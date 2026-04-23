import pandas as pd
import matplotlib.pyplot as plt

# Feedback utilities — read past feedback and user improvement hints
from utils.feedback_store import get_feedback_summary, get_improvement_hints

# LLM utility — used to interpret user feedback and decide retraining strategy
from utils.llm import get_llm_response

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Module-level state to store the latest test labels and predictions
# Used by the UI to render confusion matrix and ROC curve after training
_state = {"y_test": None, "y_pred": None}


def get_last_predictions():
    # Returns the most recent y_test and y_pred stored after training or improvement
    return _state["y_test"], _state["y_pred"]


def _build_preprocessor(X):
    # Identify numeric and categorical columns separately
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    # Apply StandardScaler to numeric columns, pass through categorical columns as-is
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", "passthrough", cat_cols)
    ])


def _compute_metrics(y_test, y_pred):
    # Calculate and return all four classification metrics as a dictionary
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred, average='weighted'),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall":    recall_score(y_test, y_pred, average='weighted', zero_division=0),
    }


def run_ml_pipeline(df: pd.DataFrame, user_req=None):
    # Assume the last column is the target variable
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    # Detect if this is a classification problem based on dtype or number of unique values
    is_classification = y.dtype == 'object' or len(y.unique()) < 20

    # Encode string labels to integers for classification
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Build the preprocessing pipeline (scaling + passthrough)
    preprocessor = _build_preprocessor(X)

    # Split data into 80% train and 20% test, stratify for classification to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if is_classification else None
    )

    # Define three model pipelines — each combines preprocessing + model in one step
    models = {
        "LogisticRegression": Pipeline([("prep", preprocessor), ("model", LogisticRegression(max_iter=300))]),
        "RandomForest":       Pipeline([("prep", preprocessor), ("model", RandomForestClassifier())]),
        "SVM":                Pipeline([("prep", preprocessor), ("model", SVC())]),
    }

    # Hyperparameter grids for each model — GridSearchCV will try all combinations
    params = {
        "LogisticRegression": {"model__C": [0.1, 1, 10]},
        "RandomForest":       {"model__n_estimators": [50, 100], "model__max_depth": [3, 5, None]},
        "SVM":                {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]},
    }

    results = {}
    best_score = 0

    # Train each model with 5-fold cross-validated grid search
    for name in models:
        grid = GridSearchCV(models[name], params[name], cv=5)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        results[name] = _compute_metrics(y_test, y_pred)

        # Track predictions of the highest accuracy model for visualization
        if results[name]["accuracy"] > best_score:
            best_score = results[name]["accuracy"]
            _state["y_test"] = y_test
            _state["y_pred"] = y_pred

    # Load past feedback to adjust model scoring
    feedback_summary = get_feedback_summary()

    # Get user-preferred metrics from UI configuration
    preferred_metrics = user_req.get("metrics", []) if user_req else []

    def score_model(model_name):
        m = results[model_name]
        # Sum scores for only the metrics the user cares about
        base = sum([
            m["accuracy"]  if "Accuracy"  in preferred_metrics else 0,
            m["f1_score"]  if "F1 Score"  in preferred_metrics else 0,
            m["precision"] if "Precision" in preferred_metrics else 0,
            m["recall"]    if "Recall"    in preferred_metrics else 0,
        ]) or m["accuracy"]  # fallback to accuracy if no metrics selected

        # Apply feedback boost/penalty — each good vote adds 0.05, each bad vote subtracts 0.05
        if model_name in feedback_summary:
            good = feedback_summary[model_name]["good"]
            bad  = feedback_summary[model_name]["bad"]
            base += (good - bad) * 0.05
        return base

    # Pick the model with the highest combined score
    best_model = max(results, key=score_model)

    # Save a bar chart comparing model accuracies to disk
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

    # Assume last column is the target
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    # Encode string labels if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Build preprocessor for the improvement run
    preprocessor = _build_preprocessor(X)

    # Split data the same way as the initial run for fair comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load feedback summary and user-written improvement hints
    feedback_summary = get_feedback_summary()
    hints = get_improvement_hints()

    # Default strategy if no hints are available
    strategy = "balanced"

    if hints:
        # Take the last 3 user suggestions and send them to the LLM
        hint_text = "; ".join(hints[-3:])
        prompt = (
            f"A user gave this feedback about an AutoML model: '{hint_text}'. "
            "Based on this, suggest ONE of these strategies: "
            "'boost_trees', 'increase_depth', 'try_gradient_boosting', 'increase_regularization'. "
            "Reply with only the strategy name."
        )
        try:
            # Ask LLM to pick the best strategy based on user's natural language feedback
            strategy = get_llm_response(prompt).strip().lower()
        except Exception:
            # Fall back to default if LLM call fails
            strategy = "boost_trees"

    # Identify models that have received more bad feedback than good — skip them
    bad_models = {
        m for m, s in feedback_summary.items()
        if s["bad"] > s["good"] + 1
    }

    # Select model and hyperparameter search space based on LLM-chosen strategy
    if "gradient_boosting" in strategy or "gradient" in strategy:
        # Use GradientBoosting — good for complex patterns and boosting weak learners
        base_model = GradientBoostingClassifier()
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        }
    elif "increase_depth" in strategy:
        # Use deeper Random Forest trees — better for complex decision boundaries
        base_model = RandomForestClassifier()
        param_dist = {
            "n_estimators": [200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }
    elif "increase_regularization" in strategy:
        # Use Logistic Regression with stronger regularization (lower C = more penalty)
        base_model = LogisticRegression(max_iter=500)
        param_dist = {"C": [0.001, 0.01, 0.1]}
    else:
        # Default: wider Random Forest search — more trees and varied depth
        base_model = RandomForestClassifier()
        param_dist = {
            "n_estimators": [200, 300, 500],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5, 10],
        }

    # Wrap model in a pipeline with preprocessing
    pipeline = Pipeline([("prep", preprocessor), ("model", base_model)])

    # Prefix param keys with "model__" to match pipeline naming convention
    prefixed = {f"model__{k}": v for k, v in param_dist.items()}

    # Use RandomizedSearchCV for faster search over a wider parameter space
    search = RandomizedSearchCV(pipeline, prefixed, n_iter=15, cv=5, random_state=42)
    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred)

    # Update global state so UI can render updated confusion matrix and ROC curve
    _state["y_test"] = y_test
    _state["y_pred"] = y_pred

    # Return improvement results including which model was used and what was skipped
    model_label = type(base_model).__name__
    return {
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "model_used": model_label,
        "strategy": strategy,
        "skipped_models": list(bad_models),
    }
