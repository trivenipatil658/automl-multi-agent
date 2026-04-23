from utils.llm import get_llm_response


def truncate(text, limit=1000):
    # Trim each agent output to 1000 characters to avoid exceeding LLM token limits
    return text[:limit]


def critic_agent(analysis, features, models, tuning, evaluation):
    # Truncate all inputs before combining — prevents the prompt from being too long
    analysis   = truncate(analysis)
    features   = truncate(features)
    models     = truncate(models)
    tuning     = truncate(tuning)
    evaluation = truncate(evaluation)

    # Combine all agent outputs into a single context block for the LLM to review
    combined = f"""
    ANALYSIS: {analysis}
    FEATURES: {features}
    MODELS: {models}
    TUNING: {tuning}
    EVALUATION: {evaluation}
    """

    # Ask the LLM to act as a critic and review the entire pipeline output
    # Kept brief (3 issues, 3 improvements, 1 suggestion) to stay within token limits
    prompt = f"""
    Review this ML pipeline briefly.

    Give:
    - 3 key issues
    - 3 improvements
    - final suggestion

    {combined}
    """

    # Send the combined prompt to the LLM and return its critique
    response = get_llm_response(prompt)

    return response
