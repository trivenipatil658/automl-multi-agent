import pandas as pd
from utils.llm import get_llm_response

def data_analyst_agent(df: pd.DataFrame):
    # Basic dataset info
    info = f"""
    Dataset Shape: {df.shape}
    Columns: {list(df.columns)}
    Missing Values:
    {df.isnull().sum().to_dict()}
    """

    # Prompt for LLM
    prompt = f"""
    You are a Data Analyst.

    Analyze this dataset information and give insights:
    {info}

    Give:
    - Data summary
    - Potential issues
    - Suggestions
    """

    response = get_llm_response(prompt)

    return response