"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
from dotenv import load_dotenv
import logging
import streamlit as st
import utils
from initialize import initialize
import components as cn
import constants as ct

############################################################
# 2. 設定関連
############################################################
st.set_page_config(page_title=ct.APP_NAME)
logger = logging.getLogger(ct.LOGGER_NAME)
load_dotenv() 

############################################################
# 2-1. データソース選択（初期化前に配置）
############################################################
st.sidebar.title("検索対象の選択")

source_option = st.sidebar.radio(
    "データソースを選択してください",
    ("ローカル文書", "Webページ")

if source_option == "ローカル文書":
    st.write(f"📂 {ct.RAG_TOP_FOLDER_PATH} 配下のファイルを読み込みます")
elif source_option == "Webページ":
    st.write(f"🌐 次のURLを読み込みます: {ct.WEB_URL_LOAD_TARGETS}")
)

############################################################
# 3. 初期化処理
############################################################
try:
    initialize()
except Exception as e:
    logger.error(f"{ct.INITIALIZE_ERROR_MESSAGE}\n{e}")
    st.error(utils.build_error_message(ct.INITIALIZE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    st.stop()

if not "initialized" in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)

############################################################
# 4. 初期表示
############################################################
cn.display_app_title()
cn.display_select_mode()
cn.display_initial_ai_message()

############################################################
# 5. 会話ログの表示
############################################################
try:
    cn.display_conversation_log()
except Exception as e:
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    st.error(utils.build_error_message(ct.CONVERSATION_LOG_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    st.stop()

############################################################
# 6. チャット入力の受け付け
############################################################
chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)

############################################################
# 7. チャット送信時の処理
############################################################
if chat_message:
    logger.info({"message": chat_message, "application_mode": st.session_state.mode})

    with st.chat_message("user"):
        st.markdown(chat_message)

    res_box = st.empty()
    with st.spinner(ct.SPINNER_TEXT):
        try:
            llm_response = utils.get_llm_response(chat_message)
        except Exception as e:
            logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
            st.error(utils.build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            st.stop()
    
    with st.chat_message("assistant"):
        try:
            if st.session_state.mode == ct.ANSWER_MODE_1:
                content = cn.display_search_llm_response(llm_response)
            elif st.session_state.mode == ct.ANSWER_MODE_2:
                content = cn.display_contact_llm_response(llm_response)
            logger.info({"message": content, "application_mode": st.session_state.mode})
        except Exception as e:
            logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
            st.error(utils.build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            st.stop()

    st.session_state.messages.append({"role": "user", "content": chat_message})
    st.session_state.messages.append({"role": "assistant", "content": content})
