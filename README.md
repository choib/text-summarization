# Korean_text_summarization

## About This Repository
This repository is dedicated to a skill of summarizing Korean text using the power of Langchain in conjunction with LLamaCpp for a local LLM server.
The purpose of this experiment is to implement text summarization by using langchain's text-split tools to make chunks of text, clustering the chunks by similarity measure, and then summarizing the cluters. This experiment is for Korean texts, and this experiment uses a Korean sentence embedding model. There are Korean sentence embedding models in huggingface.

## Prepare text from PDF
1. Install `xpdf`.
2. Preview the PDF file by using any PDF reader, and sort out unnessesary pages.
3. Adjust top and bottom margin to exclude header and footer of each page
4. Run `pdftotext` to extract text and save to other file.
`pdftotext -marginb 40 -margint 40 -nopgbrk test.pdf test.txt`

## Download large languge model and embedding model
1. Download Korean fine-tuned Mistral-7B from a huggingface page (Q4_m.GGUF file only)
`https://huggingface.co/davidkim205/komt-mistral-7b-v1-gguf/blob/main/ggml-model-q4_k_m.gguf`
2. Download a directory of Korean Sentense Embedding Model by git
`git clone https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS`

## Pull this repository from github

## Put model files and edit config.py
1. Put downloaded gguf file in `src/model`
2. Put the physical path of embedding model directory to `src/config/config.py`

## Install python 3.11

## Run App Locally
1. build environment:
`pip -r requirement.txt`

2. cd to the source directory:
`cd src`

3. run streamlit app:
`streamlit run app.py`

## Reference:
A base line of this code is from
https://github.com/yang0369/LLM_summarization