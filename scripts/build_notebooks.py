"""Generate the three project notebooks (run once; outputs are executed separately)."""
import nbformat as nbf
from pathlib import Path

HEADER = ('import sys, pathlib\n'
          'ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" '
          'else pathlib.Path.cwd()\n'
          'sys.path.insert(0, str(ROOT))\n'
          '%matplotlib inline\n'
          'import warnings; warnings.filterwarnings("ignore")')


def nb(cells, path):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_code_cell(c[1]) if c[0] == "code" else nbf.v4.new_markdown_cell(c[1])
               for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                "language": "python"}
    nbf.write(n, path)


out = Path("notebooks")
nb([
    ("md", "# 01 - Data acquisition and EDA\nHeart Disease UCI (Cleveland, 303 patients, 13 predictors)."),
    ("code", HEADER + "\nimport pandas as pd, seaborn as sns\nimport matplotlib.pyplot as plt\n"
     "from src.data import load_raw, clean\nfrom src import eda\nfrom IPython.display import Image, display\nsns.set_theme(style='whitegrid')"),
    ("md", "## 1. Load raw file and inspect\nRun `python scripts/download_data.py` first if `data/raw` is empty."),
    ("code", "raw = load_raw()\nprint(raw.shape)\nraw.head()"),
    ("code", "print('Missing values (raw, coded as ?):')\nprint(raw.isna().sum()[raw.isna().sum() > 0])\n"
     "print('\\nOriginal target distribution (0-4):')\nprint(raw['target'].value_counts().sort_index())"),
    ("md", "## 2. Cleaning\nTarget is binarised (0 = healthy, 1-4 = disease). Missing `ca`/`thal` values are kept "
     "as NaN and imputed *inside* the sklearn pipeline so imputers are fit on training folds only."),
    ("code", "df = clean(raw)\nprint(df.shape, df['target'].value_counts().to_dict())\ndf.describe().T"),
    ("md", "## 3. Class balance"),
    ("code", "eda.class_balance(df); display(Image(str(eda.FIG_DIR/'class_balance.png')))"),
    ("md", "The classes are nearly balanced (54% / 46%), so accuracy is meaningful and no resampling is needed; "
     "recall and ROC-AUC are still tracked because missing a sick patient is the costly error."),
    ("md", "## 4. Distributions of numeric features"),
    ("code", "eda.numeric_histograms(df); "
     "display(Image(str(eda.FIG_DIR/'numeric_histograms.png')))"),
    ("md", "## 5. Categorical features vs outcome"),
    ("code", "eda.categorical_counts(df); "
     "display(Image(str(eda.FIG_DIR/'categorical_counts.png')))"),
    ("md", "## 6. Correlations"),
    ("code", "eda.correlation_heatmap(df); "
     "display(Image(str(eda.FIG_DIR/'correlation_heatmap.png')))"),
    ("code", "df.corr()['target'].drop('target').sort_values(key=abs, ascending=False)"),
    ("md", "**Takeaways**: `thal`, `ca`, `oldpeak`, `exang`, `cp` and `thalach` carry most of the signal. "
     "`thalach` correlates negatively with age and with disease, which motivated the engineered "
     "`hr_ratio = thalach / (220 - age)` feature. `chol` and `fbs` are weak predictors on their own."),
    ("md", "## 7. Outliers"),
    ("code", "eda.boxplots(df); "
     "display(Image(str(eda.FIG_DIR/'boxplots.png')))"),
    ("code", "df.to_csv(ROOT/'data'/'heart_clean.csv', index=False); print('saved cleaned dataset')"),
], out / "01_eda.ipynb")

nb([
    ("md", "# 02 - Feature engineering, model selection and MLflow tracking\n"
     "Full pipeline: `FeatureEngineer -> ColumnTransformer(impute+scale | impute+one-hot) -> classifier`."),
    ("code", HEADER + "\nimport pandas as pd, mlflow, json\nfrom src.train import setup_mlflow, MODEL_ZOO\n"
     "from src.config import METRICS_PATH\nsetup_mlflow()"),
    ("md", "## Candidate models and hyper-parameter grids"),
    ("code", "for name, (est, grid) in MODEL_ZOO.items():\n    print(name, grid)"),
    ("md", "## Train (writes everything to MLflow)\nSelection uses 5-fold stratified CV ROC-AUC on the 80% "
     "training split; the 20% hold-out is only used to report final metrics."),
    ("code", "from src.train import main\nmain()"),
    ("md", "## Compare runs"),
    ("code", "runs = mlflow.search_runs(experiment_names=['heart-disease-risk'], "
     "filter_string=\"tags.stage = 'model-selection'\")\n"
     "cols = ['tags.mlflow.runName','metrics.cv_roc_auc','metrics.cv_accuracy','metrics.cv_precision',"
     "'metrics.cv_recall','metrics.test_roc_auc','metrics.test_accuracy','metrics.test_precision',"
     "'metrics.test_recall']\nruns[cols].round(3).sort_values('metrics.cv_roc_auc', ascending=False)"),
    ("code", "meta = json.loads(METRICS_PATH.read_text()); print('Selected:', meta['selected_model'])"),
    ("code", "from IPython.display import Image, display\nfor f in ['roc_logistic_regression','cm_logistic_regression',"
     "'importance_logistic_regression']:\n    display(Image(str(ROOT/'reports'/'figures'/f'{f}.png'), width=420))"),
], out / "02_training.ipynb")

nb([
    ("md", "# 03 - Inference\nLoad the packaged pipeline and score new patients, directly and through the API."),
    ("code", HEADER + "\nimport joblib, pandas as pd\nfrom src.config import MODEL_PATH, FEATURES\n"
     "model = joblib.load(MODEL_PATH)\nmodel"),
    ("code", "patients = pd.DataFrame([\n"
     "  dict(age=63,sex=1,cp=1,trestbps=145,chol=233,fbs=1,restecg=2,thalach=150,exang=0,oldpeak=2.3,slope=3,ca=0,thal=6),\n"
     "  dict(age=67,sex=1,cp=4,trestbps=160,chol=286,fbs=0,restecg=2,thalach=108,exang=1,oldpeak=1.5,slope=2,ca=3,thal=3),\n"
     "  dict(age=41,sex=0,cp=2,trestbps=130,chol=204,fbs=0,restecg=2,thalach=172,exang=0,oldpeak=1.4,slope=1,ca=0,thal=3),\n"
     "])[FEATURES]\nproba = model.predict_proba(patients)[:, 1]\n"
     "patients.assign(p_disease=proba.round(3), prediction=(proba>=.5).astype(int))"),
    ("md", "## Same predictions through the FastAPI app (in-process)"),
    ("code", "from fastapi.testclient import TestClient\nimport os\nos.environ['MODEL_PATH'] = str(MODEL_PATH)\n"
     "from api.app import app\nwith TestClient(app) as c:\n    for rec in patients.to_dict('records'):\n"
     "        print(c.post('/predict', json=rec).json())"),
], out / "03_inference.ipynb")
print("notebooks written")
