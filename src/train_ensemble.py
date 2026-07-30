from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


RANDOM_SEED = 22
NUMBER_OF_FOLDS = 5
CATBOOST_ITERATIONS = 100
LIGHTGBM_ESTIMATORS = 1000

PROJECT_DIRECTORY = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_DIRECTORY / "train.csv"
TEST_PATH = PROJECT_DIRECTORY / "test.csv"
SUBMISSION_PATH = PROJECT_DIRECTORY / "submission_oof_ensemble.csv"
WEIGHT_RESULTS_PATH = PROJECT_DIRECTORY / "oof_weight_results.csv"
DATABASE_PATH = PROJECT_DIRECTORY / "accuracyData.db"


def make_catboost_model(
    categorical_columns: tuple[str, ...],
) -> CatBoostClassifier:
    """Create a fresh CatBoost model for a fold or the final fit."""
    return CatBoostClassifier(
        iterations=CATBOOST_ITERATIONS,
        learning_rate=0.05,
        depth=7,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        cat_features=categorical_columns,
        random_seed=RANDOM_SEED,
        verbose=False,
    )


def make_lightgbm_model() -> LGBMClassifier:
    """Create a fresh LightGBM model for a fold or the final fit."""
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=LIGHTGBM_ESTIMATORS,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=4,
        verbosity=-1,
    )


def load_and_prepare_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    tuple[str, ...],
]:
    """Load the CSV files and prepare independent model inputs."""
    train_dataframe = pd.read_csv(TRAIN_PATH)
    test_dataframe = pd.read_csv(TEST_PATH)

    test_ids = test_dataframe["id"].copy()
    targets = train_dataframe["health_condition"].copy()

    raw_training = train_dataframe.drop(
        columns=["id", "health_condition"]
    ).copy()
    raw_test = test_dataframe.drop(columns=["id"]).copy()

    if list(raw_training.columns) != list(raw_test.columns):
        raise ValueError("Training and test feature columns do not match")

    categorical_columns = tuple(
        raw_training.select_dtypes(exclude="number").columns
    )

    # CatBoost accepts numeric NaN values natively. Its categorical values
    # must not be NaN, so missing categories become an explicit string value.
    catboost_training = raw_training.copy()
    catboost_test = raw_test.copy()
    for column in categorical_columns:
        catboost_training[column] = catboost_training[column].fillna("missing")
        catboost_test[column] = catboost_test[column].fillna("missing")

    # LightGBM also accepts numeric NaN values natively. Categorical columns
    # use pandas' category dtype. Test categories use the training definition;
    # a category seen only in test becomes NaN and is handled as missing.
    lightgbm_training = raw_training.copy()
    lightgbm_test = raw_test.copy()
    for column in categorical_columns:
        lightgbm_training[column] = (
            lightgbm_training[column]
            .fillna("missing")
            .astype("category")
        )
        lightgbm_test[column] = pd.Categorical(
            lightgbm_test[column].fillna("missing"),
            categories=lightgbm_training[column].cat.categories,
        )

    print(f"Loaded {len(train_dataframe):,} training rows")
    print(f"Loaded {len(test_dataframe):,} test rows")
    print(f"Features: {raw_training.shape[1]}")
    print(f"Categorical features: {len(categorical_columns)}")
    print("Target counts:")
    print(targets.value_counts())

    return (
        catboost_training,
        catboost_test,
        lightgbm_training,
        lightgbm_test,
        targets,
        test_ids,
        categorical_columns, 
    )


def create_oof_probabilities( 
    catboost_training: pd.DataFrame,
    lightgbm_training: pd.DataFrame,
    targets: pd.Series,
    categorical_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict every training row with models that did not train on it."""
    classes = np.sort(targets.unique())
    number_of_classes = len(classes)

    catboost_oof = np.zeros(
        (len(targets), number_of_classes),
        dtype=float,
    )
    lightgbm_oof = np.zeros_like(catboost_oof)

    folds = StratifiedKFold(
        n_splits=NUMBER_OF_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    for fold_number, (training_indices, validation_indices) in enumerate(
        folds.split(catboost_training, targets),
        start=1,
    ):
        print(f"\nTraining OOF fold {fold_number}/{NUMBER_OF_FOLDS}")

        fold_targets = targets.iloc[training_indices]
        catboost_fold_model = make_catboost_model(categorical_columns)
        lightgbm_fold_model = make_lightgbm_model()

        catboost_fold_model.fit(
            catboost_training.iloc[training_indices],
            fold_targets,
        )
        lightgbm_fold_model.fit(
            lightgbm_training.iloc[training_indices],
            fold_targets,
            categorical_feature=list(categorical_columns),
        )

        if not np.array_equal(catboost_fold_model.classes_, classes): #type: ignore
            raise ValueError(
                f"Unexpected CatBoost class order in fold {fold_number}"
            )
        if not np.array_equal(lightgbm_fold_model.classes_, classes):
            raise ValueError(
                f"Unexpected LightGBM class order in fold {fold_number}"
            )

        catboost_oof[validation_indices] = np.asarray(
            catboost_fold_model.predict_proba(
                catboost_training.iloc[validation_indices]
            ),
            dtype=float,
        )
        lightgbm_oof[validation_indices] = np.asarray(
            lightgbm_fold_model.predict_proba(
                lightgbm_training.iloc[validation_indices]
            ),
            dtype=float,
        )

    if np.any(catboost_oof.sum(axis=1) == 0):
        raise ValueError("Some CatBoost OOF rows were not predicted")
    if np.any(lightgbm_oof.sum(axis=1) == 0):
        raise ValueError("Some LightGBM OOF rows were not predicted")

    return catboost_oof, lightgbm_oof, classes


def find_best_weights(
    catboost_oof: np.ndarray,
    lightgbm_oof: np.ndarray,
    targets: pd.Series,
    classes: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series]:
    """Search all blend weights from 0.00 to 1.00 in 0.01 steps."""
    weight_results: list[dict[str, float]] = []

    for catboost_weight in np.linspace(0.0, 1.0, 101):
        lightgbm_weight = 1.0 - catboost_weight
        blended_probability = (
            catboost_weight * catboost_oof
            + lightgbm_weight * lightgbm_oof
        )
        blended_prediction = classes[
            blended_probability.argmax(axis=1)
        ]

        weight_results.append(
            {
                "cat_weight": float(catboost_weight),
                "lgbm_weight": float(lightgbm_weight),
                "balanced_accuracy": float(
                    balanced_accuracy_score(targets, blended_prediction)
                ),
                "accuracy": float(
                    accuracy_score(targets, blended_prediction)
                ),
                "f1_score": float(
                    f1_score(targets, blended_prediction, average="macro")
                ),
            }
        )

    results = pd.DataFrame(weight_results).sort_values(
        by="balanced_accuracy",
        ascending=False,
    )
    best_result = results.iloc[0]
    return results, best_result


def train_final_models_and_predict(
    catboost_training: pd.DataFrame,
    catboost_test: pd.DataFrame,
    lightgbm_training: pd.DataFrame,
    lightgbm_test: pd.DataFrame,
    targets: pd.Series,
    categorical_columns: tuple[str, ...],
    classes: np.ndarray,
    catboost_weight: float,
    lightgbm_weight: float,
) -> np.ndarray:
    """Train on all labeled rows and return blended test labels."""
    print("\nTraining final models on all labeled rows")
    final_catboost_model = make_catboost_model(categorical_columns)
    final_lightgbm_model = make_lightgbm_model()

    final_catboost_model.fit(catboost_training, targets)
    final_lightgbm_model.fit(
        lightgbm_training,
        targets,
        categorical_feature=list(categorical_columns),
    )

    if not np.array_equal(final_catboost_model.classes_, classes): #type:ignore
        raise ValueError("Final CatBoost class order does not match OOF order")
    if not np.array_equal(final_lightgbm_model.classes_, classes):
        raise ValueError("Final LightGBM class order does not match OOF order")

    catboost_test_probability = np.asarray(
        final_catboost_model.predict_proba(catboost_test),
        dtype=float,
    )
    lightgbm_test_probability = np.asarray(
        final_lightgbm_model.predict_proba(lightgbm_test),
        dtype=float,
    )

    if catboost_test_probability.shape != lightgbm_test_probability.shape:
        raise ValueError("Final test probability shapes do not match")

    blended_test_probability = (
        catboost_weight * catboost_test_probability
        + lightgbm_weight * lightgbm_test_probability
    )
    return classes[blended_test_probability.argmax(axis=1)]


def log_best_result(best_result: pd.Series) -> None:
    """Append the best OOF experiment to SQLite without deleting old rows."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS oofAccuracyScores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                folds INTEGER NOT NULL,
                cat_weight REAL NOT NULL,
                lgbm_weight REAL NOT NULL,
                balanced_accuracy REAL NOT NULL,
                accuracy REAL NOT NULL,
                f1_score REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO oofAccuracyScores (
                folds,
                cat_weight,
                lgbm_weight,
                balanced_accuracy,
                accuracy,
                f1_score
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                NUMBER_OF_FOLDS,
                float(best_result["cat_weight"]),
                float(best_result["lgbm_weight"]),
                float(best_result["balanced_accuracy"]),
                float(best_result["accuracy"]),
                float(best_result["f1_score"]),
            ),
        )


def main() -> None:
    (
        catboost_training,
        catboost_test,
        lightgbm_training,
        lightgbm_test,
        targets,
        test_ids,
        categorical_columns,
    ) = load_and_prepare_data()

    catboost_oof, lightgbm_oof, classes = create_oof_probabilities(
        catboost_training,
        lightgbm_training,
        targets,
        categorical_columns,
    )

    weight_results, best_result = find_best_weights(
        catboost_oof,
        lightgbm_oof,
        targets,
        classes,
    )
    weight_results.to_csv(WEIGHT_RESULTS_PATH, index=False)

    best_catboost_weight = float(best_result["cat_weight"])
    best_lightgbm_weight = float(best_result["lgbm_weight"])

    print("\nBest OOF ensemble")
    print(f"CatBoost weight: {best_catboost_weight:.2f}")
    print(f"LightGBM weight: {best_lightgbm_weight:.2f}")
    print(
        f"Balanced accuracy: {best_result['balanced_accuracy']:.6f}"
    )
    print(f"Accuracy: {best_result['accuracy']:.6f}")
    print(f"Macro F1: {best_result['f1_score']:.6f}")

    test_prediction = train_final_models_and_predict(
        catboost_training,
        catboost_test,
        lightgbm_training,
        lightgbm_test,
        targets,
        categorical_columns,
        classes,
        best_catboost_weight,
        best_lightgbm_weight,
    )

    if len(test_prediction) != len(test_ids):
        raise ValueError("Submission IDs and predictions have different lengths")

    submission = pd.DataFrame(
        {
            "id": test_ids,
            "health_condition": test_prediction,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    log_best_result(best_result)

    print(f"\nSaved submission to {SUBMISSION_PATH.name}")
    print(f"Saved weight search to {WEIGHT_RESULTS_PATH.name}")
    print(f"Logged best result to {DATABASE_PATH.name}")


if __name__ == "__main__":
    main()
