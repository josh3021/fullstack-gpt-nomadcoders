from typing import List, Literal, TypedDict

import streamlit as st
from bs4 import BeautifulSoup
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

REQUESTS_PER_SECOND = 2
FILTER_URLS = [
    r"^(.*\/ai-gateway\/).*",
    r"^(.*\/vectorize\/).*",
    r"^(.*\/workers-ai\/).*",
]


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
            
            The answers are formatted as follows:
            Answer: [answer text]
            Source: [source URL]
            Date: [date]
            
            Each answer is separated by double line breaks.
            
            Answers:
            {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


class History(TypedDict):
    question: str
    answer: str


class ChooseAnswerInputs(TypedDict):
    question: str
    answers: List[str]


class GetAnswersInputs(TypedDict):
    documents: List[BeautifulSoup]
    question: str


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ''
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message = ''
        self.message_box = st.empty()

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message + "▌")  # 커서 효과 추가

    def on_llm_end(self, *args, **kwargs):
        if self.message_box:
            self.message_box.markdown(self.message)  # 최종 메시지에서 커서 제거


class SiteGPT():
    def __init__(self, url: str):
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
        documents = loader.load_and_split(text_splitter=splitter)
        vector_store = FAISS.from_documents(documents, OpenAIEmbeddings())
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

    def choose_answer(self, inputs: ChooseAnswerInputs):
        question = inputs['question']
        answers = inputs['answers']

        # 스트리밍을 위한 새로운 콜백 핸들러 생성
        callback_handler = ChatCallbackHandler()
        llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[callback_handler]
        )

        choose_chain = choose_prompt | llm

        # 줄바꿈을 명확하게 처리하기 위해 다양한 방법 시도
        answer_blocks = []
        for i, answer in enumerate(answers, 1):
            # 각 답변 블록을 구분하기 위해 번호와 구분선 추가
            answer_block = f"""
                --- Answer {i} ---
                Answer: {answer['answer']}
                Source: {answer['source']}
                Date: {answer['date']}
                ---"""
            answer_blocks.append(answer_block)

        # 답변들을 명확하게 구분하여 결합
        condensed_answer = '\n\n'.join(answer_blocks)

        # 추가 처리: 줄바꿈 문자 정규화
        condensed_answer = condensed_answer.replace('\\n', '\n')

        result = choose_chain.invoke({
            "question": question,
            "answers": condensed_answer
        })

        return result

    def get_answers(self, inputs: GetAnswersInputs):
        docs = inputs['documents']
        question = inputs['question']

        # 각 문서에 대해 개별적으로 처리하되 스트리밍은 최종 답변에서만
        llm = ChatOpenAI(
            temperature=0.1,
            streaming=False  # 중간 처리는 스트리밍 없이
        )
        answers_chain = answers_prompt | llm

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

    def invoke_chain(self, question: str):
        chain = {
            "documents": self.retriever,
            "question": RunnablePassthrough(),
        } | RunnableLambda(self.get_answers) | RunnableLambda(self.choose_answer)
        result = chain.invoke(question)
        return result.content.replace("$", "\$")

    def save_message(self, message: str, role: Literal['user', 'assistant', 'ai', 'human']):
        st.session_state['messages'].append({"message": message, "role": role})

    def send_message(self, message: str, role: Literal['user', 'assistant', 'ai', 'human'], save=True):
        with st.chat_message(role):
            st.markdown(message)
        if save:
            self.save_message(message, role)

    def ask_to_llm(self, question):
        self.send_message(question, 'user', True)

        # AI 응답을 위한 채팅 메시지 컨테이너 생성
        with st.chat_message('assistant'):
            # 스트리밍 응답 생성
            ai_message = self.invoke_chain(question)
            # 최종 메시지는 이미 스트리밍으로 출력됨

        # 세션 상태에 최종 메시지 저장
        self.save_message(ai_message, 'assistant')

    def paint_history(self):
        for message in st.session_state['messages']:
            self.send_message(
                message['message'],
                message['role'],
                save=False
            )


st.set_page_config(
    page_title="SiteGPT",
    layout="wide",
    page_icon="🖥️",
)


with st.sidebar:
    url = st.text_input('Write down a URL', placeholder="https://example.com")


st.markdown(
    """
    # SiteGPT

    Ask questions about the content of a website.

    Start by writing the URL of the website on the sidebar.
    """
)

if url:
    if not url.endswith(".xml"):
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        sitegpt = SiteGPT(url)
        sitegpt.paint_history()
        question = st.chat_input("Ask a question to the website!")
        if question:
            sitegpt.ask_to_llm(question)
else:
    st.session_state['messages'] = []
