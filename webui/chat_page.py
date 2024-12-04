import streamlit as st
from utils import PLATFORMS, get_models, get_chatllm

def get_chat_response(platform, model, temperature, input):
    llm = get_chatllm(platform, model, temperature=temperature)
    for chunk in llm.stream(input):
        yield chunk.content

def display_chat_history():
    for message in st.session_state["chat_history"]:
        with st.chat_message(message[0]):
            st.write(message[1])

def clear_chat_history():
    st.session_state["chat_history"] = []


def chat_page():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    with st.sidebar:
        st.title("Chatbot")


    display_chat_history()

    with st._bottom:
        cols = st.columns([1.2, 10, 1])
        with cols[0].popover(":gear:", use_container_width=True, help="配置模型"):
            platform = st.selectbox("请选择要使用的模型加载方式", PLATFORMS)
            model = st.selectbox("请选择要使用的模型", get_models(platform))
            temperature = st.slider("请选择历史消息长度", 0.1, 1., 0.1)
            history_len = st.slider("请选择历史消息长度", 1, 10, 5)
        input = cols[1].chat_input("请输入您的问题")
        cols[2].button(":wastebasket:", help="清空对话", on_click=clear_chat_history)
    if input:
        with st.chat_message("user"):
            st.write(input)
        st.session_state["chat_history"] += [('user', input)]

        stream_response = get_chat_response(
            platform,
            model,
            temperature,
            st.session_state["chat_history"][-history_len:]
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream_response)
        st.session_state["chat_history"] += [('assistant', response)]