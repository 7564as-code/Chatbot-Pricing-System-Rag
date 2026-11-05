"""
קליינט פשוט לבדיקת הבוט, בנוי עם Streamlit.

הרצה:
    pip install streamlit
    streamlit run app.py

דורש קובץ rag_engine.py עם הפונקציה:
    ask(question: str, history: list[dict] | None) -> {"answer": str, "sources": [...]}
"""

import streamlit as st

from rag_engine import ask

st.set_page_config(page_title="בוט ידע ארגוני", page_icon="💬", layout="centered")

# עיצוב מימין לשמאל לעברית
st.markdown(
    """
    <style>
    .stApp, .stChatMessage, .stMarkdown, .stExpander {
        direction: rtl;
        text-align: right;
    }
    textarea { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💬 בוט ידע ארגוני")
st.caption("הבוט עונה רק על סמך המסמכים של הארגון.")

# סרגל צד: אפשרויות בדיקה
with st.sidebar:
    st.header("הגדרות בדיקה")
    show_sources = st.toggle("הצג מקורות וציונים", value=True)
    if st.button("נקה שיחה"):
        st.session_state.messages = []
        st.rerun()

# היסטוריית השיחה נשמרת ב-session_state
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    """מציג את הצ'אנקים שהבוט השתמש בהם."""
    if not sources:
        return
    with st.expander("מקורות"):
        for s in sources:
            st.markdown(
                f"- **{s.get('doc', '')}** | סעיף: {s.get('section', '')} "
                f"| רמה: {s.get('level', '')} | ציון: {s.get('score', 0):.3f}"
            )


# הצגת ההודעות הקודמות
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg["role"] == "assistant":
            render_sources(msg.get("sources"))

# קלט חדש מהמשתמש
question = st.chat_input("שאלי שאלה על המסמכים...")

if question:
    # ההיסטוריה שנשלחת לבוט היא לפני ההודעה הנוכחית
    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("מחפש בתשובות..."):
            try:
                result = ask(question, history)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
            except Exception as e:  # הצגת שגיאה ברורה במקום קריסה
                answer = f"אירעה שגיאה: {e}"
                sources = []

        st.markdown(answer)
        if show_sources:
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
