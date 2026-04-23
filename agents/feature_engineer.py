import pandas as pd
from utils.llm import get_llm_response


def feature_engineer_agent(df: pd.DataFrame):
    # Collect column names, data types, and a small sample to give the LLM context
    info = f"""
    Columns: {list(df.columns)}
    Data Types:
    {df.dtypes.to_dict()}
    Sample Data:
    {df.head(5).to_dict()}
    """

    # Ask the LLM to suggest feature engineering steps based on the dataset structure
    prompt = f"""
    You are a Feature Engineering Expert.

    Analyze this dataset and suggest:
    - Feature transformations
    - Encoding techniques
    - Scaling methods
    - Feature selection ideas

    Dataset info:
    {info}
    """

    # Send the prompt to the LLM and return its feature engineering suggestions
    response = get_llm_response(prompt)

    return response
