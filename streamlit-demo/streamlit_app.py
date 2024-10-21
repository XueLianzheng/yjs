import os

import streamlit as st

from streamlit_gallery.utils.page import page_group

#使用Streamlit构建的Web应用程序，展示项目功能的演示。
#这段代码实现了一个基于 Streamlit 的用户界面，旨在展示一个 LLM（大语言模型）画廊应用。
#通过这个应用，用户可以与多个不同的聊天组件和功能进行交互。

def main():
    from streamlit_gallery.apps import gallery #gallery 用于显示 LLM 聊天画廊
    from streamlit_gallery.components import chat, doc_chat #chat 和 doc_chat 组件用于实现聊天功能。

    # 设置侧边栏
    page = page_group("p")#创建一个名为 page_group 的对象 page，用于管理页面内容。
    with st.sidebar:
        st.title("🎉 LLM Gallery")#在侧边栏中设置标题为 "🎉 LLM Gallery"。

        #添加应用和组件的选择
        with st.expander("✨ APPS", True):#使用 st.expander 创建可折叠的菜单项，包含 “✨ APPS”。
            page.item("LLM Chat Gallery", gallery, default=True)#默认显示的应用是 "LLM Chat Gallery"。

        #添加其他聊天组件
        with st.expander("🧩 COMPONENTS", True):

            if os.getenv("CHAT_API_BASE", ""):#通过 os.getenv 检查环境变量的存在性。
                page.item("Chat", chat)
                page.item("Doc Chat", doc_chat)
            #检查环境变量 CHAT_API_BASE 是否存在，如果存在，则添加 "Chat" 和 "Doc Chat" 组件

            #添加其他组件(根据不同的环境变量，动态加载和显示组件。)
            if os.getenv("SQL_CHAT_API_BASE", ""):# SQL 聊天
                from streamlit_gallery.components import sql_chat
                page.item("SQL Chat", sql_chat)

            if os.getenv("SERPAPI_API_KEY", ""):#搜索聊天
                from streamlit_gallery.components import search_chat
                page.item("Search Chat", search_chat)

            if os.getenv("TOOL_CHAT_API_BASE", ""):#工具聊天
                from streamlit_gallery.components import tool_chat
                page.item("Tool Chat", tool_chat)

            if os.getenv("INTERPRETER_CHAT_API_BASE", ""):#代码解释器
                from streamlit_gallery.components import code_interpreter
                page.item("Code Interpreter", code_interpreter)

            from streamlit_gallery.components import multimodal_chat#多模态聊天
            page.item("Multimodal Chat", multimodal_chat)

        #清空消息按钮
        if st.button("🗑️ 清空消息"):
            st.session_state.messages = []
        #添加一个按钮，用户点击后会清空当前会话中的消息。

        #模型配置
        with st.expander("✨ 模型配置", False):#创建一个可折叠的面板，用于配置模型。
            model_name = st.text_input(label="模型名称")
            base_url = st.text_input(label="模型接口地址", value=os.getenv("CHAT_API_BASE"))
            api_key = st.text_input(label="API KEY", value=os.getenv("API_KEY", "xxx"))
            #提供文本输入框，让用户输入模型名称、接口地址和 API 密钥。

            st.session_state.update(#将这些配置保存到 st.session_state，以便在会话中使用。
                dict(
                    model_name=model_name,
                    base_url=base_url,
                    api_key=api_key,
                )
            )

        #参数配置
        with st.expander("🐧 参数配置", False):#添加另一个可折叠的面板，让用户配置参数。
            max_tokens = st.slider("回复最大token数量", 20, 4096, 1024)
            temperature = st.slider("温度", 0.0, 1.0, 0.9)
            chunk_size = st.slider("文档分块大小", 100, 512, 250)
            chunk_overlap = st.slider("文档分块重复大小", 0, 100, 50)
            top_k = st.slider("文档分块检索数量", 0, 10, 4)
            #提供了多个滑块（slider），用户可以设置回复的最大 token 数量、温度、文档分块大小、分块重叠大小和检索数量。

            st.session_state.update(#将参数值更新到 st.session_state。
                dict(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                )
            )

    page.show()#显示页面,最后，调用 page.show() 显示构建的页面。


if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit LLM Gallery", page_icon="🎈", layout="wide")
    main()
    #在脚本直接运行时，设置页面配置（标题、图标、布局），并调用 main() 函数。
