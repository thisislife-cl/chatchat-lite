import streamlit as st
from chat_page import chat_page
from agent_chat_page import agent_chat_page
from rag_chat_page import rag_chat_page

if __name__ == "__main__":
    # with st.sidebar:
    #     st.logo("img.png", size="large")
    pg = st.navigation({
        "对话": [
            st.Page(chat_page, title="对话", icon=":material/chat_bubble:"),
            st.Page(rag_chat_page, title="RAG 对话", icon=":material/chat:"),
            st.Page(agent_chat_page, title="Agent 对话", icon=":material/chat_add_on:"),
        ],
        "设置": [
            st.Page(st.write, title="知识库管理", icon=":material/library_books:"),
            st.Page(st.chat_input, title="模型管理", icon=":material/settings:"),
        ]

    })
    pg.run()