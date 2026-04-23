from langgraph.graph import StateGraph

from agents.data_analyst import data_analyst_agent
from agents.feature_engineer import feature_engineer_agent
from agents.model_selector import model_selection_agent
from agents.hyperparameter_tuner import hyperparameter_tuning_agent
from agents.evaluator import evaluation_agent
from agents.critic import critic_agent


# Define state
from typing import TypedDict
import pandas as pd

class AgentState(TypedDict, total=False):
    df: pd.DataFrame
    analysis: str
    features: str
    models: str
    tuning: str
    evaluation: str
    critic: str

def data_analyst_node(state):
    state["analysis"] = data_analyst_agent(state["df"])
    return state


def feature_engineer_node(state):
    state["features"] = feature_engineer_agent(state["df"])
    return state


def model_selector_node(state):
    state["models"] = model_selection_agent(state["df"])
    return state


def tuner_node(state):
    state["tuning"] = hyperparameter_tuning_agent(state["df"])
    return state


def evaluator_node(state):
    state["evaluation"] = evaluation_agent(state["df"])
    return state


def critic_node(state):
    state["critic"] = critic_agent(
        state["analysis"],
        state["features"],
        state["models"],
        state["tuning"],
        state["evaluation"]
    )
    return state


def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("data_analyst", data_analyst_node)
    graph.add_node("feature_engineer", feature_engineer_node)
    graph.add_node("model_selector", model_selector_node)
    graph.add_node("tuner", tuner_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("critic", critic_node)

    # Flow
    graph.set_entry_point("data_analyst")

    graph.add_edge("data_analyst", "feature_engineer")
    graph.add_edge("feature_engineer", "model_selector")
    graph.add_edge("model_selector", "tuner")
    graph.add_edge("tuner", "evaluator")
    graph.add_edge("evaluator", "critic")

    return graph.compile()