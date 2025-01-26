import streamlit as st
from helper import pdf_read,get_chunks,get_vectordb,get_chain
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(
    page_title='Custom PDF Assistant',
    page_icon="imgs/avatar_streamly.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Multi PDF Assistance")

if "proceed" not in st.session_state:
    st.session_state.proceed = False


with st.sidebar:
    st.subheader("Upload PDF's here")
    pdfs = st.file_uploader("Browse the PDF's and click Proceed",accept_multiple_files=True)
    if st.button("Proceed"):
        with st.spinner("Processing"):
            text = pdf_read(pdfs)
        st.success("PDF's Readed Successfully")
        with st.spinner("Creating Vector DataBase"):
            chunks = get_chunks(text)
            vectordb = get_vectordb(chunks)
            st.session_state.chain = get_chain(vectordb)
        st.success("Now you can use the chatbot ")
        st.session_state.proceed=True

if st.session_state.proceed==False:
    st.info('Please Upload a PDF First')
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content='Hello, I am a PDF Assistant. How can I help you?')
        ]
    
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = st.session_state.chain({'question': user_query})
        st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if len(st.session_state.chat_history) == 1:
            with st.chat_message("AI",avatar='imgs/ai.png'):
                st.write(message.content)
        elif i % 2 == 0:
            with st.chat_message("Human",avatar='imgs/human.png'):
                st.write(message.content)
        else:
            with st.chat_message("AI",avatar='imgs/ai.png'):
                st.write(message.content)
