import json
import os

FILE_PATH = "feedback.json"


def load_feedback():
    if not os.path.exists(FILE_PATH):
        return []
    with open(FILE_PATH, "r") as f:
        return json.load(f)


def save_feedback(entry):
    data = load_feedback()
    data.append(entry)
    with open(FILE_PATH, "w") as f:
        json.dump(data, f, indent=4)


def get_feedback_summary():
    data = load_feedback()
    summary = {}
    for entry in data:
        if "model" not in entry:
            continue
        model = entry["model"]
        feedback = entry.get("feedback", "")
        if model not in summary:
            summary[model] = {"good": 0, "bad": 0}
        if "👍" in feedback or "Accurate" in feedback:
            summary[model]["good"] += 1
        else:
            summary[model]["bad"] += 1
    return summary


def get_improvement_hints():
    """Return list of user-written suggestion texts from negative feedback."""
    data = load_feedback()
    hints = []
    for entry in data:
        suggestion = entry.get("suggestion", "").strip()
        feedback = entry.get("feedback", "")
        is_bad = "👎" in feedback or "Inaccurate" in feedback
        if is_bad and suggestion:
            hints.append(suggestion)
    return hints
