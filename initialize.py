"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from docx import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    logger = logging.getLogger(ct.LOGGER_NAME)
    if logger.hasHandlers():
        return

    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )

    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []


def initialize_retriever():
    """
    RAGのRetrieverを作成（選択したデータソースのみ読み込み）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    if "retriever" in st.session_state:
        return

    # 選択したデータソースに応じて読み込む
    source_option = st.session_state.get("source_option", "ローカル文書")

    docs_all = []

    if source_option in ["ローカル文書", "両方"]:
        # ローカル文書の読み込み
        docs_all.extend(load_local_data_sources())

    if source_option in ["Webページ", "両方"]:
        # Webページの読み込み
        docs_all.extend(load_web_data_sources())

    # OSがWindowsの場合、文字列調整
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    # 埋め込みモデル
    embeddings = OpenAIEmbeddings()
    
    #問題6で差し替え
    # 🔹 社員名簿はチャンク分割せず、それ以外は分割
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n"
    )

    splitted_docs = []
    for doc in docs_all:
        filename = doc.metadata.get("source", "")
        if "社員名簿" in filename:  # 🔹社員名簿ならそのまま追加
            splitted_docs.append(doc)
        else:  # それ以外はチャンク分割
            splitted_docs.extend(text_splitter.split_documents([doc]))
    
    # ベクターストア作成
    db = Chroma.from_documents(splitted_docs, embedding=embeddings)
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RETRIEVER_TOP_K})


def load_local_data_sources():
    """
    ローカル文書の読み込み
    """
    docs_all = []
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)
    return docs_all


def load_web_data_sources():
    """
    Webページの読み込み
    """
    docs_all = []
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        loader = WebBaseLoader(web_url)
        docs_all.extend(loader.load())
    return docs_all


def recursive_file_check(path, docs_all):
    """
    ディレクトリ再帰的にファイルを読み込み
    """
    if not os.path.exists(path):
        return

    if os.path.isdir(path):
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            recursive_file_check(full_path, docs_all)
    else:
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイルをロード
    """
    file_extension = os.path.splitext(path)[1]
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs_all.extend(loader.load())


def adjust_string(s):
    """
    Windows用文字列調整
    """
    if type(s) is not str:
        return s
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
    return s
