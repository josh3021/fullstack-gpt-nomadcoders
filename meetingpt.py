import math
import os
import subprocess
from glob import glob

import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydub import AudioSegment
from streamlit.runtime.uploaded_file_manager import UploadedFile

HAS_TRANSCRIPT = os.path.exists("./.cache/podcast.txt")


@st.cache_data()
def write_video(video: UploadedFile):
    video_content = video.read()
    with open(f"./.cache/{video.name}", "wb") as f:
        f.write(video_content)


@st.cache_data()
def extract_audio_from_video(video_path: str):
    if HAS_TRANSCRIPT:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path: str, chunk_size: int, chunks_folder: str):
    if HAS_TRANSCRIPT:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if HAS_TRANSCRIPT:
        return
    files = glob(f"{chunk_folder}/*.mp3")
    sorted_files = sorted(files)
    for file in sorted_files:
        with open(file, 'rb') as audio_file, open(destination, 'a') as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            text_file.write(transcript['text'])


st.set_page_config(
    page_title="MeetinGPT",
    page_icon="💼"
)

st.markdown(
    """
    # MeetingGPT

    Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

    Get started by uploading a video file in the sidebar.
    """
)

llm = ChatOpenAI(
    temperature=0.1
)

with st.sidebar:
    video_input = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])

if video_input:
    CHUNKS_FOLDER = "./.cache/chunks"
    VIDEO_PATH = f"./.cache/{video_input.name}"
    AUDIO_PATH = VIDEO_PATH.replace("mp4", "mp3")
    TRANSCRIPT_PATH = VIDEO_PATH.replace("mp4", "txt")
    with st.status("Loading Video...") as status:
        write_video(video_input)
        status.update(label="Extracting Audio...")
        extract_audio_from_video(VIDEO_PATH)
        status.update(label="Cutting Audio Segments...")
        cut_audio_in_chunks(AUDIO_PATH, 10, CHUNKS_FOLDER)
        status.update(label="Transcribing Audio...")
        transcribe_chunks(CHUNKS_FOLDER, TRANSCRIPT_PATH)

    transcript_tab, summary_tab, qna_tab = st.tabs(
        ["Transcript", "Summary", "Q&A"]
    )

    with transcript_tab:
        with open(file=TRANSCRIPT_PATH, mode="r", encoding='UTF-8') as file:
            st.write(file.read())

    with summary_tab:
        start = st.button("Generate Summary")

        if start:
            loader = TextLoader(TRANSCRIPT_PATH)
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800,
                chunk_overlap=100,
            )
            documents = loader.load_and_split(text_splitter=splitter)
            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:
                """
            )

            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            with st.status("Summarizing") as status:
                documents_len = len(documents)
                status.update(
                    label=f"Processing document 1/{documents_len}...")
                summary = first_summary_chain.invoke({
                    "text": documents[0].page_content
                })
                refined_prompt = ChatPromptTemplate.from_template(
                    """
                    Your job is to produce a final summary.
                    We have provided an existing summary up to a certain point: {existing_summary}
                    We have the opportunity to refine the existing summary (only if needed) with some more context below.
                    ------------
                    {context}
                    ------------
                    Given the new context, refine the original summary.
                    If the context isn't useful, RETURN the original summary.
                    """
                )
                refined_chain = refined_prompt | llm | StrOutputParser()

                for idx, document in enumerate(documents[1:]):
                    status.update(
                        label=f"Processing document {idx+2}/{documents_len}...")
                    summary = refined_chain.invoke({
                        "existing_summary": summary,
                        "context": document.page_content
                    })
            st.write(summary)

    # TODO complete QNA Tab
    with qna_tab:
        pass
