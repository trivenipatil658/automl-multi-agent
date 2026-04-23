import pandas as pd
from utils.llm import get_llm_response


def model_selection_agent(df: pd.DataFrame):
    # Provide column names, data types, and the assumed target variable to the LLM
    info = f"""
    Columns: {list(df.columns)}
    Data Types:
    {df.dtypes.to_dict()}
    Target Variable: Assume last column is target -> {df.columns[-1]}
    """

    # Ask the LLM to recommend suitable ML models based on the dataset characteristics
    prompt = f"""
    You are an ML Expert.

    Based on the dataset info, suggest:
    - Problem type (classification/regression)
    - Best ML models to try
    - Why those models are suitable

    Dataset:
    {info}
    """

    # Send the prompt to the LLM and return its model recommendations
    response = get_llm_response(prompt)

    return response
