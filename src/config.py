"""Central configuration: paths, column definitions and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "processed.cleveland.data"
CLEAN_PATH = DATA_DIR / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "heart_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FIG_DIR = ROOT / "reports" / "figures"
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = ROOT / "mlartifacts"
EXPERIMENT_NAME = "heart-disease-risk"
REGISTERED_MODEL_NAME = "heart-disease-classifier"

# Column order of the UCI "processed.cleveland.data" file
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Discrete / coded features -> one-hot encoded
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
# Engineered in src.preprocessing.FeatureEngineer
ENGINEERED_FEATURES = ["hr_ratio"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "1 = male, 0 = female",
    "cp": "Chest pain type (1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG result (0 normal, 1 ST-T abnormality, 2 LV hypertrophy)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of peak exercise ST segment (1 up, 2 flat, 3 down)",
    "ca": "Number of major vessels (0-3) coloured by fluoroscopy",
    "thal": "Thalassemia (3 normal, 6 fixed defect, 7 reversible defect)",
}
