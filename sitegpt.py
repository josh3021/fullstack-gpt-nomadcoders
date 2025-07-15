from typing import List, TypedDict

import streamlit as st
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

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

REQUESTS_PER_SECOND = 2

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question.
    If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.
    The score should be high if the answer is related to the user's question, and low otherwise.
    If there is no relevant content, the score is 0.
    Always provide scores with your answers
    
    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!
    Question: {question}
    """
)


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            
            AND MOSTLY IMPORTANT THING IS DETERMITE THE QUESTION'S LANGAUGE AND ANSWER WITH THE SAME LANGUAGE!
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


class History(TypedDict):
    question: str
    answer: str


class Inputs(TypedDict):
    question: str
    answers: List[str]


class GET_ANSWERS_INPUTS(TypedDict):
    documents: List[BeautifulSoup]
    question: str


FILTER_URLS = [
    r"^(.*\/ai-gateway\/).*",
    r"^(.*\/vectorize\/).*",
    r"^(.*\/workers-ai\/).*",
]


class SiteGPT():
    def __init__(self, url: str):
        self.llm = ChatOpenAI(
            temperature=0.1
        )
        self.url = url
        self.history: List[History] = []
        self.retriever = self._load_website(url)

    @staticmethod
    @st.cache_data(show_spinner='Loading Website...')
    def _load_website(url: str):
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200
        )
        loader = SitemapLoader(
            web_path=url,
            filter_urls=FILTER_URLS,
            parsing_function=SiteGPT._parse_page,
        )
        loader.requests_per_second = REQUESTS_PER_SECOND
        docs = loader.load_and_split(text_splitter=splitter)
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        return vector_store.as_retriever()

    @staticmethod
    def _parse_page(soup: BeautifulSoup):
        header = soup.find('header')
        footer = soup.find('footer')
        if header:
            header.decompose()
        if footer:
            footer.decompose()

        return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ").replace('CloudflareSign upLanguagesEnglishEnglish (United Kingdom)DeutschEspañol (Latinoamérica)Español (España)FrançaisItaliano日本語한국어PolskiPortuguês (Brasil)Русский繁體中文简体中文Platform', '')

    def choose_answer(self, inputs: Inputs):
        question = inputs['question']
        answers = inputs['answers']
        choose_chain = choose_prompt | self.llm
        condensed_answer = '\n\n'.join(
            f"Answer: {answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\n"
            for answer in answers
        )
        return choose_chain.invoke({
            "question": question,
            "answers": condensed_answer
        })

    def get_answers(self, inputs: GET_ANSWERS_INPUTS):
        docs = inputs['documents']
        question = inputs['question']
        answers_chain = answers_prompt | self.llm

        return {
            "question": question,
            "answers": [
                {
                    "answer": answers_chain.invoke({
                        "question": question,
                        "context": doc.page_content
                    }).content,
                    "source": doc.metadata['source'],
                    'date': doc.metadata['lastmod']
                } for doc in docs
            ]
        }

    def invoke_chain(self, query: str):
        chain = {
            "documents": self.retriever,
            "question": RunnablePassthrough(),
        } | RunnableLambda(self.get_answers) | RunnableLambda(self.choose_answer)
        result = chain.invoke(query)
        return result.content.replace("$", "\$")


with st.sidebar:
    url = st.text_input('Write down a URL', placeholder="https://example.com")

if url:
    if not url.endswith(".xml"):
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        sitegpt = SiteGPT(url)
        query = st.text_input("Ask a question to the website!")
        if query:
            result = sitegpt.invoke_chain(query)
            st.write(result)
