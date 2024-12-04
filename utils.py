
from typing import Literal
from langchain_openai import ChatOpenAI
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout


PLATFORMS = ["ollama", "xinference", "fastchat", "openai"]


def get_models(platform_type: Literal[tuple(PLATFORMS)]):
    if platform_type == "ollama":
        import ollama
        models = [model["model"] for model in ollama.list()["models"]]
        return models
    elif platform_type == "xinference":
        from xinference_client import Client
        client = Client()
        models = client.list_models()
        return models


def get_chatllm(
        platform_type: Literal[tuple(PLATFORMS)],
        model: str,
        temperature: float = 0.9
):
    if platform_type == "ollama":
        # from langchain_ollama import ChatOllama
        # return ChatOllama
        return ChatOpenAI(
            temperature=temperature,
            model_name=model,
            streaming=True,
            base_url="http://127.0.0.1:11434/v1",
            api_key="EMPTY",
        )
    elif platform_type == "xinference":
        from langchain_community.llms import Xinference
        return Xinference


def show_graph(graph):
    flow_state = StreamlitFlowState(
                       nodes=[StreamlitFlowNode(
                           id=node.id,
                           pos=(0,0),
                           data={"content": node.id},
                           node_type="input" if node.id == "__start__"
                                             else "output" if node.id == "__end__"
                                             else "default",
                       ) for node in graph.nodes.values()],
                       edges=[StreamlitFlowEdge(
                           id=str(enum),
                           source=edge.source,
                           target=edge.target,
                           animated=True,
                       ) for enum, edge in enumerate(graph.edges)],
                   )
    streamlit_flow('example_flow',
                   flow_state,
                   layout=TreeLayout(direction='down'), fit_view=True
    )