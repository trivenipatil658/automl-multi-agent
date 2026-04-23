import pandas as pd
from utils.llm import get_llm_response

def feature_engineer_agent(df: pd.DataFrame):
    info = f"""
    Columns: {list(df.columns)}
    Data Types:
    {df.dtypes.to_dict()}
    Sample Data:
    {df.head(5).to_dict()}
    """

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

    response = get_llm_response(prompt)

    return response