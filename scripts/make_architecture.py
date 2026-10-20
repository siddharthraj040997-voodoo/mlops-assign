"""Render the architecture diagram to report/architecture.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

fig, ax = plt.subplots(figsize=(14, 8.2))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8.2)
ax.axis("off")

COL = {"data": "#e9c46a", "ml": "#2a9d8f", "ci": "#264653", "run": "#e76f51",
       "mon": "#8d6cab"}


def box(x, y, w, h, text, kind, fs=9.5, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=COL[kind], ec="white", lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold", linespacing=1.3)


def arrow(x1, y1, x2, y2, label=None, dy=0.12):
    ax.annotate("", (x2, y2), (x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.6))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center", fontsize=8,
                color="#333")


def lane(y, h, title):
    ax.add_patch(FancyBboxPatch((0.15, y), 13.7, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#f6f6f6", ec="#cccccc", lw=1))
    ax.text(0.3, y + h - 0.22, title, fontsize=10, color="#555", fontweight="bold")


ax.text(7, 7.95, "Heart Disease MLOps - Architecture", ha="center", fontsize=15,
        fontweight="bold")

lane(5.55, 2.1, "1  DATA & MODEL DEVELOPMENT")
box(0.5, 5.8, 2.3, 1.2, "UCI Heart Disease\n(Cleveland, 303 rows)\ndownload_data.py", "data")
box(3.3, 5.8, 2.3, 1.2, "Cleaning + EDA\nsrc/data.py\nsrc/eda.py", "data")
box(6.1, 5.8, 2.6, 1.2, "sklearn Pipeline\nFeatureEngineer ->\nImpute/Scale/OneHot -> Clf", "ml")
box(9.2, 5.8, 2.1, 1.2, "Train + GridSearchCV\nLR / RF / GB\nsrc/train.py", "ml")
box(11.7, 5.8, 1.9, 1.2, "MLflow\nparams, metrics,\nplots, model", "ml")
arrow(2.8, 6.4, 3.3, 6.4); arrow(5.6, 6.4, 6.1, 6.4)
arrow(8.7, 6.4, 9.2, 6.4); arrow(11.3, 6.4, 11.7, 6.4)

lane(3.35, 2.0, "2  CI/CD  (GitHub Actions)")
box(0.5, 3.6, 2.2, 1.2, "git push / PR", "ci")
box(3.2, 3.6, 2.2, 1.2, "flake8 lint", "ci")
box(5.9, 3.6, 2.2, 1.2, "pytest\n(data, model, API)", "ci")
box(8.6, 3.6, 2.2, 1.2, "Train + verify\nupload artifacts", "ci")
box(11.3, 3.6, 2.3, 1.2, "docker build\n+ smoke test", "ci")
for a, b in [(2.7, 3.2), (5.4, 5.9), (8.1, 8.6), (10.8, 11.3)]:
    arrow(a, 4.2, b, 4.2)

lane(0.2, 3.0, "3  SERVING, DEPLOYMENT & MONITORING")
box(0.5, 1.75, 2.4, 1.1, "Docker image\npython:3.12-slim\n+ model.joblib", "run")
box(3.5, 1.75, 2.6, 1.1, "Kubernetes\nDeployment (2 pods)\nHPA + probes", "run")
box(6.7, 1.75, 2.2, 1.1, "Service\nLoadBalancer / Ingress", "run")
box(9.5, 1.75, 1.9, 1.1, "Client\nPOST /predict", "data", tc="#222")
box(0.5, 0.4, 2.4, 0.9, "JSON request logs\n(stdout / kubectl logs)", "mon", fs=9)
box(3.5, 0.4, 2.6, 0.9, "FastAPI\n/predict /health /metrics", "run")
box(6.7, 0.4, 2.2, 0.9, "Prometheus", "mon")
box(9.5, 0.4, 1.9, 0.9, "Grafana\ndashboard", "mon")
arrow(2.9, 2.3, 3.5, 2.3); arrow(6.1, 2.3, 6.7, 2.3)
arrow(9.5, 2.3, 8.9, 2.3)
arrow(4.8, 1.75, 4.8, 1.3, "serves")
arrow(3.5, 0.85, 2.9, 0.85); arrow(6.1, 0.85, 6.7, 0.85, "scrape /metrics", dy=0.15)
arrow(8.9, 0.85, 9.5, 0.85)
arrow(12.6, 5.8, 12.5, 4.8, "model.joblib")
fig.savefig("report/architecture.png", dpi=160, bbox_inches="tight", facecolor="white")
print("saved report/architecture.png")
