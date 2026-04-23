import pandas as pd
from utils.llm import get_llm_response

def hyperparameter_tuning_agent(df: pd.DataFrame):
    info = f"""
    Columns: {list(df.columns)}
    Target Variable: {df.columns[-1]}
    """

    prompt = f"""
    You are an ML Optimization Expert.

    Suggest:
    - Important hyperparameters for common ML models
    - Tuning methods (Grid Search, Random Search, etc.)
    - Best practices for improving performance

    Dataset:
    {info}
    """

    response = get_llm_response(prompt)

    return response