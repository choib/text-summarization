from io import StringIO

import streamlit as st
from langchain.docstore.document import Document
# from langchain.embeddings import VertexAIEmbeddings
from streamlit.components.v1 import html
from load_and_chunk import ProcessingPipeline
from summarize_long_mistral import summarize_long_text_by_custom #, summarize_long_text_by_langchain
import time
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from config import config

import os
os.environ['TOKENIZERS_PARALLELISM']='true'
embedding_local = HuggingFaceEmbeddings(
                    model_name=config.EMBED_PATH,
                    encode_kwargs={"normalize_embeddings": True},
                )

st.title('텍스트 요약하기')

#input_text = st.text_area('Please paste the text you want to summarise below')
input_text = ""
uploaded_file = st.file_uploader("PDF문서를 텍스트로 변환한 후, 사용해주십시오.Choose a file (supported type: .txt)")
if uploaded_file is not None:
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # To read file as string:
    input_text = stringio.read()
#print(input_text)
with st.container():
    st.write(f'Number Of Words: {len(input_text.split(" "))}')

if_summarise = st.button("요약하기", type="primary")

if if_summarise and len(input_text) > 0:
    with st.container():
        start = time.time()
        while len(input_text.split(" ")) > config.CHUNK_SIZE:
        #if len(input_text.split(" ")) > 500:
            pro = ProcessingPipeline(embedding_local) 
            chunks = pro.process_document(input_text)
            split_docs = [Document(page_content=text, metadata={"source": "local"}) for text in chunks]
            input_text, key_idea = summarize_long_text_by_custom(split_docs)
        
        summary = input_text
        end = time.time()

        st.divider()
        st.header('요약:', divider='rainbow')
        st.write(f'{summary}')
        st.write(f'Number Of Words: {len(summary.split(" "))}')

        st.header('맥락:', divider='rainbow')
        st.write(f'{key_idea}')
        st.write(f'Number Of Words: {len(key_idea.split(" "))}')
        st.write(f'Total Time Taken: {end - start} seconds')
