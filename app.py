import os

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.base import RunnableLambda
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class Memory:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)

    def save_memory(self, input_text, output_text):
        self.memory.save_context(
            {"input": input_text},
            {"output": output_text}
        )

    def load_memory_variables(self):
        return self.memory.load_memory_variables({})

    def get_chat_history(self):
        """채팅 히스토리를 문자열로 반환"""
        memorized_messages = self.memory.chat_memory.messages
        history = []
        for memorized_message in memorized_messages:
            if isinstance(memorized_message, HumanMessage):
                history.append(f"Human: {memorized_message.content}")
            elif isinstance(memorized_message, AIMessage):
                history.append(f"AI: {memorized_message.content}")
        return "\n".join(history)


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ''
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
        self.message = ""

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        if self.message_box is not None:
            self.message_box.markdown(self.message)

    def on_llm_end(self, *args, **kwargs):
        # 스트리밍이 끝나면 메시지 저장
        save_message(self.message, 'ai')


def send_message(msg_content, role, save=True):
    with st.chat_message(role):
        st.markdown(msg_content)
    if save:
        save_message(msg_content, role)


def save_message(msg_content, role):
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    st.session_state['messages'].append({"message": msg_content, "role": role})


def paint_history():
    if 'messages' in st.session_state:
        for message in st.session_state['messages']:
            send_message(message['message'], message['role'], False)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(original_file):
    os.makedirs("./.cache/files", exist_ok=True)
    file_content = original_file.read()
    file_path = f"./.cache/files/{original_file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{original_file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    # 문서 로드 및 분할
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    # 임베딩 생성
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)

    # 벡터 스토어 생성
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# 메인 UI
st.title("DocumentGPT")

st.markdown(
    """
    Welcome!
                
    Use this chatbot to ask questions to an AI about your files!
    
    Upload your files on the sidebar.
    """
)

with st.sidebar:
    api_key_input = st.text_input(
        'OpenAI API키를 입력해주세요.',
        type="password",
        help="sk-로 시작하는 OpenAI API 키를 입력하세요."
    )

    # 파일 업로드
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

    st.link_button('Go to Github Repository',
                   "https://github.com/josh3021/fullstack-gpt-nomadcoders")

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state['messages'] = []


if api_key_input and file:
    try:
        # 파일이나 API 키가 변경되면 메시지 초기화
        if 'current_file' not in st.session_state or st.session_state.get('current_file') != file:
            st.session_state['messages'] = []
            st.session_state['current_file'] = file

        # 파일 임베딩
        retriever = embed_file(file)

        # 준비 완료 메시지 (한 번만 표시)
        if len(st.session_state['messages']) == 0:
            send_message("파일이 준비되었습니다! 파일에 대해 무엇이든 물어보세요...", "ai", False)

        # 채팅 기록 표시
        paint_history()

        # 사용자 입력
        user_input = st.chat_input("파일에 대해 무엇이든 물어보세요...")

        if user_input:
            # 사용자 메시지 저장 및 표시
            send_message(user_input, 'human')

            # 콜백 핸들러 생성 (매번 새로 생성)
            callback_handler = ChatCallbackHandler()

            # LLM 초기화
            llm = ChatOpenAI(
                api_key=api_key_input,
                model="gpt-4o-mini",
                temperature=0.1,
                streaming=True,
                callbacks=[callback_handler]
            )

            # 프롬프트 템플릿
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 """
                    Answer the question using ONLY the following context.
                    If you don't know the answer just say you don't know.
                    Don't make anything up.
                    And most important thing is ANSWER ONLY IN KOREAN!

                    Context: {context}
                """),
                ("human", "{question}")
            ])

            # 체인 생성
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
            )

            # AI 응답 생성 (스트리밍)
            with st.chat_message('ai'):
                response = chain.invoke(user_input)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
        st.info("API 키가 올바른지 확인해주세요.")

elif not api_key_input:
    st.info("👈 사이드바에서 OpenAI API 키를 입력해주세요.")

elif not file:
    st.info("👈 사이드바에서 파일을 업로드해주세요.")
