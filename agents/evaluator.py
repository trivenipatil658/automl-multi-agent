import pandas as pd
from utils.llm import get_llm_response

def evaluation_agent(df: pd.DataFrame):
    info = f"""
    Target Column: {df.columns[-1]}
    """

    prompt = f"""
    You are an ML Evaluation Expert.

    Suggest:
    - Best evaluation metrics for this problem
    - How to validate the model (train/test split, cross-validation)
    - How to compare multiple models

    Dataset info:
    {info}
    """

    response = get_llm_response(prompt)

    return response