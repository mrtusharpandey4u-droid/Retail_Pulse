import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load

DATA_PATH = "retail_sales_dataset.csv"
MODEL_PATH = "retail_pulse_model.joblib"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    df["day_of_week"] = df["Date"].dt.day_name().fillna("Unknown")
    df["month"] = df["Date"].dt.month
    feature_columns = ["Gender", "Age", "Quantity", "Price per Unit", "day_of_week", "month"]
    target_column = "Product Category"
    return df[feature_columns], df[target_column]


def build_pipeline() -> Pipeline:
    numeric_features = ["Age", "Quantity", "Price per Unit", "month"]
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_features = ["Gender", "day_of_week"]
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    return pipeline


def train_and_save(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    df = load_data(data_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    dump(pipeline, model_path)
    return pipeline, metrics


def load_model(model_path: str = MODEL_PATH) -> Pipeline:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train_model.py first.")
    return load(model_path)


def build_input_row(gender: str, age: int, quantity: int, price_per_unit: float, date: str = None) -> dict:
    row = {
        "Gender": gender,
        "Age": int(age),
        "Quantity": int(quantity),
        "Price per Unit": float(price_per_unit),
    }
    if date:
        dt = pd.to_datetime(date, errors="coerce")
        row["day_of_week"] = dt.day_name() if not pd.isna(dt) else "Unknown"
        row["month"] = int(dt.month) if not pd.isna(dt) else pd.NA
    else:
        row["day_of_week"] = "Unknown"
        row["month"] = pd.NA
    return row


def predict_category(model: Pipeline, payload: dict) -> str:
    samples = pd.DataFrame([payload])
    prediction = model.predict(samples)
    return str(prediction[0])
