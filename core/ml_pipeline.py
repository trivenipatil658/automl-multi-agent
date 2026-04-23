import pandas as pd
import matplotlib.pyplot as plt
from utils.feedback_store import get_feedback_summary, get_improvement_hints
from utils.llm import get_llm_response

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

_state = {"y_test": None, "y_pred": None}


def get_last_predictions():
    return _state["y_test"], _state["y_pred"]


def _build_preprocessor(X):
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", "passthrough", cat_cols)
    ])


def _compute_metrics(y_test, y_pred):
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred, average='weighted'),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall":    recall_score(y_test, y_pred, average='weighted', zero_division=0),
    }


def run_ml_pipeline(df: pd.DataFrame, user_req=None):
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    is_classification = y.dtype == 'object' or len(y.unique()) < 20
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    preprocessor = _build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if is_classification else None
    )

    models = {
        "LogisticRegression": Pipeline([("prep", preprocessor), ("model", LogisticRegression(max_iter=300))]),
        "RandomForest":       Pipeline([("prep", preprocessor), ("model", RandomForestClassifier())]),
        "SVM":                Pipeline([("prep", preprocessor), ("model", SVC())]),
    }

    params = {
        "LogisticRegression": {"model__C": [0.1, 1, 10]},
        "RandomForest":       {"model__n_estimators": [50, 100], "model__max_depth": [3, 5, None]},
        "SVM":                {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]},
    }

    results = {}
    best_score = 0

    for name in models:
        grid = GridSearchCV(models[name], params[name], cv=5)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        results[name] = _compute_metrics(y_test, y_pred)
        if results[name]["accuracy"] > best_score:
            best_score = results[name]["accuracy"]
            _state["y_test"] = y_test
            _state["y_pred"] = y_pred

    feedback_summary = get_feedback_summary()
    preferred_metrics = user_req.get("metrics", []) if user_req else []

    def score_model(model_name):
        m = results[model_name]
        base = sum([
            m["accuracy"]  if "Accuracy"  in preferred_metrics else 0,
            m["f1_score"]  if "F1 Score"  in preferred_metrics else 0,
            m["precision"] if "Precision" in preferred_metrics else 0,
            m["recall"]    if "Recall"    in preferred_metrics else 0,
        ]) or m["accuracy"]
        if model_name in feedback_summary:
            good = feedback_summary[model_name]["good"]
            bad  = feedback_summary[model_name]["bad"]
            base += (good - bad) * 0.05   # meaningful penalty/boost
        return base

    best_model = max(results, key=score_model)

    plt.figure()
    model_names = list(results.keys())
    plt.bar(model_names, [results[m]["accuracy"] for m in model_names])
    plt.title("Model Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()

    return {"results": results, "best_model": best_model}


def improve_model(df: pd.DataFrame):
    """Retrain with wider search, guided by human feedback hints via LLM."""
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    preprocessor = _build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    feedback_summary = get_feedback_summary()
    hints = get_improvement_hints()

    # Ask LLM what to try based on user hints
    strategy = "balanced"
    if hints:
        hint_text = "; ".join(hints[-3:])   # last 3 suggestions
        prompt = (
            f"A user gave this feedback about an AutoML model: '{hint_text}'. "
            "Based on this, suggest ONE of these strategies: "
            "'boost_trees', 'increase_depth', 'try_gradient_boosting', 'increase_regularization'. "
            "Reply with only the strategy name."
        )
        try:
            strategy = get_llm_response(prompt).strip().lower()
        except Exception:
            strategy = "boost_trees"

    # Penalise models with bad feedback — skip them if heavily penalised
    bad_models = {
        m for m, s in feedback_summary.items()
        if s["bad"] > s["good"] + 1
    }

    # Choose model and params based on LLM strategy
    if "gradient_boosting" in strategy or "gradient" in strategy:
        base_model = GradientBoostingClassifier()
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        }
    elif "increase_depth" in strategy:
        base_model = RandomForestClassifier()
        param_dist = {
            "n_estimators": [200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }
    elif "increase_regularization" in strategy:
        base_model = LogisticRegression(max_iter=500)
        param_dist = {"C": [0.001, 0.01, 0.1]}
    else:  # boost_trees / default
        base_model = RandomForestClassifier()
        param_dist = {
            "n_estimators": [200, 300, 500],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5, 10],
        }

    pipeline = Pipeline([("prep", preprocessor), ("model", base_model)])
    prefixed = {f"model__{k}": v for k, v in param_dist.items()}

    search = RandomizedSearchCV(pipeline, prefixed, n_iter=15, cv=5, random_state=42)
    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred)

    _state["y_test"] = y_test
    _state["y_pred"] = y_pred

    model_label = type(base_model).__name__
    return {
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "model_used": model_label,
        "strategy": strategy,
        "skipped_models": list(bad_models),
    }
