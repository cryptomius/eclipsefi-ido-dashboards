import streamlit as st

st.set_page_config(
    page_title="Eclipse Fi Dashboard",
    page_icon="🌓",
    layout="wide"
)

st.title("Welcome to Eclipse Fi Dashboard")

st.markdown("""
This dashboard provides various insights into Eclipse Fi projects:

- **📊 Participation**: View participation metrics across projects
- **🌊 Cosmic Essence Flow**: Analyze user tier movements between projects
- **📈 Analytics**: General analytics and statistics

Select a page from the sidebar to get started!
""")