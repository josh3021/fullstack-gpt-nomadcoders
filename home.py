"""Module creating a home by Streamlit."""
import streamlit as st

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🤖",
)

st.markdown(
    """
# Hello!
            
Welcome to my FullstackGPT Portfolio!
            
Here are the apps I made:

- [x] [DocumentGPT](/document_gpt)
- [ ] [PrivateGPT](/private_gpt)
- [ ] [QuizGPT](/quiz_gpt)
- [ ] [SiteGPT](/site_gpt)
- [ ] [MeetingGPT](/meeting_gpt)
- [ ] [InvestorGPT](/investor_gpt)
"""
)
