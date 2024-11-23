
from typing import Literal
from langchain_openai import ChatOpenAI

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
