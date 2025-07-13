import os
import re

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.output_parsers import RegexParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.base import RunnableLambda
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

st.set_page_config(
    page_title="DocumentQuiz",
    page_icon="📚",
)


class QuizCallbackHandler(BaseCallbackHandler):
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
        pass


@st.cache_data(show_spinner="Embedding file...")
def embed_file(original_file, api_key_input):
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
    embeddings = OpenAIEmbeddings(openai_api_key=api_key_input)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)

    # 벡터 스토어 생성
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever, docs


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def parse_quiz_response(response_text):
    """Parse the quiz response using regex patterns"""

    questions = []

    # Split by question markers
    question_blocks = re.split(r'문제\s*\d+:', response_text)

    for block in question_blocks[1:]:  # Skip first empty split
        try:
            # Extract question text
            question_match = re.search(
                r'^([^A-D]*?)(?=A\)|선택지)', block.strip(), re.MULTILINE | re.DOTALL)
            if not question_match:
                continue

            question = question_match.group(1).strip()

            # Extract options
            option_a = re.search(r'A\)\s*([^\n]*(?:\n(?!B\))[^\n]*)*)', block)
            option_b = re.search(r'B\)\s*([^\n]*(?:\n(?!C\))[^\n]*)*)', block)
            option_c = re.search(r'C\)\s*([^\n]*(?:\n(?!D\))[^\n]*)*)', block)
            option_d = re.search(
                r'D\)\s*([^\n]*(?:\n(?!정답|해설)[^\n]*)*)', block)

            # Extract correct answer
            answer_match = re.search(r'정답[:\s]*([A-D])', block)

            # Extract explanation
            explanation_match = re.search(
                r'(?:해설|설명)[:\s]*(.+?)(?=문제\s*\d+:|$)', block, re.DOTALL)

            if all([question, option_a, option_b, option_c, option_d, answer_match]):
                questions.append({
                    'question': question,
                    'option_a': option_a.group(1).strip(),
                    'option_b': option_b.group(1).strip(),
                    'option_c': option_c.group(1).strip(),
                    'option_d': option_d.group(1).strip(),
                    'correct_answer': answer_match.group(1),
                    'explanation': explanation_match.group(1).strip() if explanation_match else "설명이 없습니다."
                })
        except Exception as e:
            continue

    return questions


@st.cache_data(show_spinner="Generating quiz...")
def generate_quiz(full_content, api_key_input, difficulty, num_questions, _file_hash):
    """Generate quiz using structured prompts with caching"""

    # Create LLM
    llm = ChatOpenAI(
        api_key=api_key_input,
        model="gpt-4o-mini",
        temperature=0.1,
    )

    # Create the quiz generation prompt
    difficulty_instructions = {
        "easy": "기본적인 내용 이해와 사실 확인을 위한 쉬운 문제들을 만드세요.",
        "medium": "개념 이해와 관계 파악이 필요한 보통 수준의 문제들을 만드세요.",
        "hard": "비판적 사고와 분석이 필요한 어려운 문제들을 만드세요."
    }

    # Create example format
    example_format = """
문제 1: 여기에 문제를 작성하세요
A) 첫 번째 선택지
B) 두 번째 선택지  
C) 세 번째 선택지
D) 네 번째 선택지
정답: A
해설: 정답에 대한 자세한 설명을 작성하세요.

문제 2: 다음 문제를 작성하세요
A) 첫 번째 선택지
B) 두 번째 선택지
C) 세 번째 선택지  
D) 네 번째 선택지
정답: B
해설: 정답에 대한 자세한 설명을 작성하세요.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
        당신은 퀴즈 생성기입니다. 주어진 문서 내용을 바탕으로 {num_questions}개의 객관식 문제를 만드세요.
        
        난이도: {difficulty}
        지침: {difficulty_instructions.get(difficulty, difficulty_instructions["medium"])}
        
        다음 형식을 정확히 따라서 작성하세요:
        {example_format}
        
        필수 요구사항:
        1. 모든 문제와 선택지는 한국어로 작성
        2. 각 문제는 정확히 4개의 선택지 (A, B, C, D)
        3. "문제 X:" 형식으로 문제 번호 표시
        4. 선택지는 "A) 내용" 형식으로 작성
        5. "정답: X" 형식으로 정답 표시
        6. "해설: 내용" 형식으로 설명 작성
        7. 각 문제 사이에 빈 줄 추가
        8. 문서 내용을 정확히 반영한 문제 작성
        
        반드시 위의 형식을 정확히 따라주세요.
        """),
        ("human", "문서 내용: {content}")
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({"content": full_content})
        questions = parse_quiz_response(response.content)

        if len(questions) == 0:
            st.error("퀴즈 파싱에 실패했습니다. 다시 시도해주세요.")
            with st.expander("원본 응답 보기"):
                st.text(response.content)
            return None

        return {"questions": questions}
    except Exception as e:
        st.error(f"퀴즈 생성 중 오류가 발생했습니다: {str(e)}")
        return None


def display_quiz(quiz_data):
    """Display the quiz and handle user answers"""

    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}

    questions = quiz_data["questions"]

    st.subheader("📝 퀴즈")

    # Display questions
    for i, question in enumerate(questions):
        st.markdown(f"**문제 {i+1}:** {question['question']}")

        # Create options list
        options = [
            f"A) {question['option_a']}",
            f"B) {question['option_b']}",
            f"C) {question['option_c']}",
            f"D) {question['option_d']}"
        ]

        # Radio buttons for options
        user_answer = st.radio(
            f"답을 선택하세요 (문제 {i+1})",
            options,
            key=f"question_{i}",
            index=None
        )

        if user_answer:
            # Store just the letter (A, B, C, D)
            st.session_state.user_answers[i] = user_answer[0]

        st.markdown("---")

    # Submit button
    if st.button("답안 제출", type="primary"):
        if len(st.session_state.user_answers) == len(questions):
            show_results(quiz_data)
        else:
            st.warning("모든 문제에 답해주세요!")


def show_results(quiz_data):
    """Show quiz results and handle retakes"""

    questions = quiz_data["questions"]
    correct_count = 0

    st.subheader("📊 결과")

    # Check answers
    for i, question in enumerate(questions):
        user_answer = st.session_state.user_answers.get(i)
        correct_answer = question['correct_answer']

        if user_answer == correct_answer:
            correct_count += 1
            st.success(f"✅ 문제 {i+1}: 정답! ({correct_answer})")
        else:
            st.error(
                f"❌ 문제 {i+1}: 오답 (정답: {correct_answer}, 선택: {user_answer})")
            st.info(f"💡 해설: {question['explanation']}")

    # Show final score
    score_percentage = (correct_count / len(questions)) * 100
    st.metric(
        "점수", f"{correct_count}/{len(questions)} ({score_percentage:.1f}%)")

    # Perfect score celebration
    if correct_count == len(questions):
        st.success("🎉 완벽합니다! 모든 문제를 맞췄습니다!")
        st.balloons()
    else:
        st.info(f"💪 {len(questions) - correct_count}개 문제를 더 맞춰보세요!")

        # Retake button
        if st.button("다시 시도하기", type="secondary"):
            st.session_state.user_answers = {}
            st.session_state.quiz_generated = False
            st.rerun()


# Main UI
st.title("📚 DocumentQuiz")

st.markdown(
    """
    업로드한 문서를 바탕으로 퀴즈를 생성하고 풀어보세요!
    
    사이드바에서 파일을 업로드하고 퀴즈 설정을 조정하세요.
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

    st.markdown("### 퀴즈 설정")

    # Quiz difficulty
    difficulty = st.selectbox(
        "난이도 선택",
        ["easy", "medium", "hard"],
        format_func=lambda x: {"easy": "쉬움", "medium": "보통", "hard": "어려움"}[x]
    )

    # Number of questions
    num_questions = st.slider(
        "문제 수",
        min_value=3,
        max_value=10,
        value=5
    )

    st.markdown("### 난이도 설명")
    if difficulty == "easy":
        st.info("📖 속독하면 풀 수 있어요!")
    elif difficulty == "medium":
        st.info("🤔 정독해야지 풀 수 있어요!")
    else:
        st.info("🧠 여러번 정독해봐야할 거에요!")

    st.link_button('Go to Github Repository',
                   "https://github.com/josh3021/fullstack-gpt-nomadcoders")

# Initialize session state
if 'quiz_generated' not in st.session_state:
    st.session_state.quiz_generated = False
if 'current_quiz' not in st.session_state:
    st.session_state.current_quiz = None

if api_key_input and file:
    try:
        # Check if file changed
        if 'current_file' not in st.session_state or st.session_state.get('current_file') != file:
            st.session_state.current_file = file
            st.session_state.quiz_generated = False
            st.session_state.current_quiz = None
            if 'user_answers' in st.session_state:
                del st.session_state.user_answers

        # Embed file
        retriever, docs = embed_file(file, api_key_input)
        full_content = format_docs(docs)

        st.success("✅ 파일이 성공적으로 업로드되었습니다!")

        # Generate quiz button
        if not st.session_state.quiz_generated:
            if st.button("퀴즈 생성하기", type="primary"):
                # Create hash for caching
                file_hash = hash(file.name + str(file.size) +
                                 difficulty + str(num_questions))

                quiz_data = generate_quiz(
                    full_content, api_key_input, difficulty, num_questions, file_hash)
                if quiz_data:
                    st.session_state.current_quiz = quiz_data
                    st.session_state.quiz_generated = True
                    st.rerun()

        # Display quiz if generated
        if st.session_state.quiz_generated and st.session_state.current_quiz:
            display_quiz(st.session_state.current_quiz)

            # New quiz button
            if st.button("새 퀴즈 생성", type="secondary"):
                st.session_state.quiz_generated = False
                st.session_state.current_quiz = None
                if 'user_answers' in st.session_state:
                    del st.session_state.user_answers
                st.rerun()

    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
        st.info("API 키가 올바른지 확인해주세요.")

elif not api_key_input:
    st.info("👈 사이드바에서 OpenAI API 키를 입력해주세요.")

elif not file:
    st.info("👈 사이드바에서 파일을 업로드해주세요.")
