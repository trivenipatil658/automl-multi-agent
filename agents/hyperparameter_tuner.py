import pandas as pd
from utils.llm import get_llm_response


def hyperparameter_tuning_agent(df: pd.DataFrame):
    # Pass column names and target variable info to give the LLM context about the problem
    info = f"""
    Columns: {list(df.columns)}
    Target Variable: {df.columns[-1]}
    """

    # Ask the LLM to suggest hyperparameters and tuning strategies for common ML models
    prompt = f"""
    You are an ML Optimization Expert.

    Suggest:
    - Important hyperparameters for common ML models
    - Tuning methods (Grid Search, Random Search, etc.)
    - Best practices for improving performance

    Dataset:
    {info}
    """

    # Send the prompt to the LLM and return its hyperparameter tuning suggestions
    response = get_llm_response(prompt)

    return response
