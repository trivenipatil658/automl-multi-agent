import pandas as pd
from utils.llm import get_llm_response


def evaluation_agent(df: pd.DataFrame):
    # Pass the target column name so the LLM knows what kind of problem this is
    info = f"""
    Target Column: {df.columns[-1]}
    """

    # Ask the LLM to recommend evaluation metrics and validation strategies
    prompt = f"""
    You are an ML Evaluation Expert.

    Suggest:
    - Best evaluation metrics for this problem
    - How to validate the model (train/test split, cross-validation)
    - How to compare multiple models

    Dataset info:
    {info}
    """

    # Send the prompt to the LLM and return its evaluation recommendations
    response = get_llm_response(prompt)

    return response
