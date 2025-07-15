import streamlit as st
from langchain.document_loaders import SitemapLoader

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
    """
)


# class Site():
#     def __init__(self, url: str):
#         # async chromium loader
#         self.loader = AsyncChromiumLoader([url])
#         self.docs = self.loader.load()
#         html2text_transformer = Html2TextTransformer()
#         self.transformed_docs = html2text_transformer.transform_documents(
#             self.docs)

#     def get_docs(self):
#         return self.transformed_docs

@st.cache_data(show_spinner='Loading Website...')
def load_website(url):
    loader = SitemapLoader(url)
    loader.requests_per_second = 5
    docs = loader.load()
    return docs


with st.sidebar:
    url = st.text_input('Write down a URL', placeholder="https://example.com")

if url:
    if not url.endswith(".xml"):
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        docs = load_website(url)
