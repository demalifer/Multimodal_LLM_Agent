"""Simple Streamlit UI for multimodal QA.

Features:
- Enter a question
- Upload an image
- Call backend API
- Display answer
"""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st

st.set_page_config(page_title="Multimodal QA Demo", page_icon="🖼️", layout="centered")

st.title("🖼️ 多模态问答 Demo")
st.caption("输入问题 + 上传图片，调用后端 API 并显示回答。")

api_url = st.text_input("后端 API 地址", value="http://127.0.0.1:8000/chat")
question = st.text_area("请输入问题", placeholder="例如：这张图里有什么？", height=120)
uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="已上传图片", use_container_width=True)


def _extract_answer(payload: Any) -> str:
    """Extract answer text from common response schemas."""
    if isinstance(payload, dict):
        for key in ("answer", "response", "result", "output"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("answer", "response", "result", "output", "message"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    if isinstance(payload, str) and payload.strip():
        return payload
    return "未在响应中找到可展示的回答字段。"


if st.button("提交", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("请先输入问题。")
    elif uploaded_file is None:
        st.warning("请先上传图片。")
    elif not api_url.strip():
        st.warning("请填写后端 API 地址。")
    else:
        with st.spinner("正在调用后端 API..."):
            try:
                file_bytes = uploaded_file.getvalue()
                files = {
                    "image": (uploaded_file.name, file_bytes, uploaded_file.type or "application/octet-stream"),
                }
                data = {"question": question.strip()}
                response = requests.post(api_url.strip(), data=data, files=files, timeout=60)
                response.raise_for_status()

                try:
                    payload = response.json()
                except ValueError:
                    payload = response.text

                st.success("调用成功")
                st.subheader("回答")
                st.write(_extract_answer(payload))

                with st.expander("查看原始响应"):
                    st.code(str(payload), language="json")
            except requests.RequestException as exc:
                st.error(f"调用后端失败：{exc}")
