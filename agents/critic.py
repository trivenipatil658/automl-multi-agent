from utils.llm import get_llm_response


def truncate(text, limit=1000):
    return text[:limit]


def critic_agent(analysis, features, models, tuning, evaluation):
    # 🔥 Reduce size (IMPORTANT)
    analysis = truncate(analysis)
    features = truncate(features)
    models = truncate(models)
    tuning = truncate(tuning)
    evaluation = truncate(evaluation)

    combined = f"""
    ANALYSIS: {analysis}
    FEATURES: {features}
    MODELS: {models}
    TUNING: {tuning}
    EVALUATION: {evaluation}
    """

    prompt = f"""
    Review this ML pipeline briefly.

    Give:
    - 3 key issues
    - 3 improvements
    - final suggestion

    {combined}
    """

    response = get_llm_response(prompt)

    return response