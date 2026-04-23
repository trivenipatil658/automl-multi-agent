import json
import os

# Path to the JSON file where all feedback entries are persisted
FILE_PATH = "feedback.json"


def load_feedback():
    # Return empty list if no feedback file exists yet
    if not os.path.exists(FILE_PATH):
        return []
    # Load and return all feedback entries from the JSON file
    with open(FILE_PATH, "r") as f:
        return json.load(f)


def save_feedback(entry):
    # Load existing feedback, append the new entry, and write back to file
    data = load_feedback()
    data.append(entry)
    with open(FILE_PATH, "w") as f:
        json.dump(data, f, indent=4)


def get_feedback_summary():
    # Build a per-model summary of good vs bad feedback counts
    data = load_feedback()
    summary = {}

    for entry in data:
        # Skip entries that don't have a model field
        if "model" not in entry:
            continue

        model = entry["model"]
        feedback = entry.get("feedback", "")

        # Initialize counters for this model if seen for the first time
        if model not in summary:
            summary[model] = {"good": 0, "bad": 0}

        # Count as good if thumbs up or "Accurate" appears in the feedback text
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
        # Get the user's typed suggestion (may be empty string)
        suggestion = entry.get("suggestion", "").strip()
        feedback = entry.get("feedback", "")

        # Only collect suggestions that came with negative feedback
        is_bad = "👎" in feedback or "Inaccurate" in feedback

        # Only add non-empty suggestions from negative feedback entries
        if is_bad and suggestion:
            hints.append(suggestion)

    return hints
