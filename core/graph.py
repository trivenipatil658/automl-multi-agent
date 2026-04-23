# LangGraph — used to build a stateful multi-agent pipeline as a directed graph
from langgraph.graph import StateGraph

# Import each specialized agent function
from agents.data_analyst import data_analyst_agent
from agents.feature_engineer import feature_engineer_agent
from agents.model_selector import model_selection_agent
from agents.hyperparameter_tuner import hyperparameter_tuning_agent
from agents.evaluator import evaluation_agent
from agents.critic import critic_agent

from typing import TypedDict
import pandas as pd


# AgentState defines the shared state passed between all agent nodes in the graph
# Each key holds the output string from the corresponding agent
# total=False means all keys are optional — not all need to be set at once
class AgentState(TypedDict, total=False):
    df: pd.DataFrame   # the input dataset shared across all agents
    analysis: str      # output from data analyst agent
    features: str      # output from feature engineer agent
    models: str        # output from model selector agent
    tuning: str        # output from hyperparameter tuner agent
    evaluation: str    # output from evaluator agent
    critic: str        # final review output from critic agent


# Each node function receives the current state, calls its agent, updates state, and returns it

def data_analyst_node(state):
    # Run the data analyst agent — analyses dataset shape, columns, missing values
    state["analysis"] = data_analyst_agent(state["df"])
    return state


def feature_engineer_node(state):
    # Run the feature engineer agent — suggests transformations, encoding, scaling
    state["features"] = feature_engineer_agent(state["df"])
    return state


def model_selector_node(state):
    # Run the model selector agent — recommends suitable ML models for the problem
    state["models"] = model_selection_agent(state["df"])
    return state


def tuner_node(state):
    # Run the hyperparameter tuner agent — suggests tuning strategies and best params
    state["tuning"] = hyperparameter_tuning_agent(state["df"])
    return state


def evaluator_node(state):
    # Run the evaluator agent — recommends evaluation metrics and validation strategies
    state["evaluation"] = evaluation_agent(state["df"])
    return state


def critic_node(state):
    # Run the critic agent — reviews all previous agent outputs and gives final feedback
    state["critic"] = critic_agent(
        state["analysis"],
        state["features"],
        state["models"],
        state["tuning"],
        state["evaluation"]
    )
    return state


def build_graph():
    # Create a new LangGraph state machine using AgentState as the shared state schema
    graph = StateGraph(AgentState)

    # Register each agent as a named node in the graph
    graph.add_node("data_analyst", data_analyst_node)
    graph.add_node("feature_engineer", feature_engineer_node)
    graph.add_node("model_selector", model_selector_node)
    graph.add_node("tuner", tuner_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("critic", critic_node)

    # Set the first node to run when the graph is invoked
    graph.set_entry_point("data_analyst")

    # Define the sequential flow — each agent passes its output to the next
    graph.add_edge("data_analyst", "feature_engineer")
    graph.add_edge("feature_engineer", "model_selector")
    graph.add_edge("model_selector", "tuner")
    graph.add_edge("tuner", "evaluator")
    graph.add_edge("evaluator", "critic")

    # Compile the graph into a runnable app
    return graph.compile()
