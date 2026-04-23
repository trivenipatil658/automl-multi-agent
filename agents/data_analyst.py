import pandas as pd
from utils.llm import get_llm_response


def data_analyst_agent(df: pd.DataFrame):
    # Gather basic structural information about the dataset
    info = f"""
    Dataset Shape: {df.shape}
    Columns: {list(df.columns)}
    Missing Values:
    {df.isnull().sum().to_dict()}
    """

    # Build a prompt asking the LLM to act as a data analyst and review the dataset
    prompt = f"""
    You are a Data Analyst.

    Analyze this dataset information and give insights:
    {info}

    Give:
    - Data summary
    - Potential issues
    - Suggestions
    """

    # Send the prompt to the LLM and return its analysis
    response = get_llm_response(prompt)

    return response
