'''
rule-book에 대해 처리하는 코드. PDF 파일 로드 - 텍스트, 이미지 및 테이블 추출 - 파일로 저장하는 과정으로 이루어진다.
'''
import os
import argparse
import tiktoken
import subprocess
import uuid
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ['MKL_SERVICE_FORCE_INTEL']='1'
os.environ['MKL_THREADING_LAYER']='GNU'

def tiktoken_len(text):   # 대부분의 LLM은 토큰 단위로 데이터를 입력받음 
    tokenizer = tiktoken.get_encoding('cl100k_base')
    # tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = tokenizer.encode(text)
    return len(tokens)

def cleanse_text(text):
    '''
    다중 줄바꿈 제거 및 특수 문자 중복 제거
    '''
    import re 
    text = re.sub(r'(\n\s*)+\n+', '\n\n', text)
    text = re.sub(r"\·{1,}", " ", text)
    text = re.sub(r"\.{1,}", ".", text)
    return text

def main(args):
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    file_name = args.file_name 
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tok_chunk_size = 500
    loader = PyPDFLoader(os.path.join(args.document_path, args.file_name))
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=tok_chunk_size,
        chunk_overlap=20,
        length_function=tiktoken_len   # else len (character)
    )
    print(pages[0].metadata)   # source (filepath, page)

    txt_list = []; source_list = []; page_list =[]; chunk_size_list = []
    for p_idx in range(len(pages)):
        texts = text_splitter.create_documents([pages[p_idx].page_content])
        print(f'len of texts: {len(texts)}')
        
        for idx in range(len(texts)):
            txt = texts[idx].page_content
            if idx == 0:
                # print(txt)
                pass
            cleansed_txt = cleanse_text(txt)
            txt_list.append(cleansed_txt)
            source_list.append(pages[p_idx].metadata['source'])
            page_list.append(pages[p_idx].metadata['page'])
            chunk_size_list.append(tok_chunk_size)
           
    data_df = pd.DataFrame(zip(txt_list, source_list, page_list, chunk_size_list), columns=['text', 'source', 'page_no', 'chunk_size'])
    data_df.to_csv(os.path.join(args.output_dir, f'{args.file_name.split(".")[0]}.csv'))

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--document_path', type=str, default='../data/pdf')
    cli_parser.add_argument('--file_name', type=str, default='취업규칙.pdf')
    cli_parser.add_argument('--output_dir', type=str, default='../data/output')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)