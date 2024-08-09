from pymilvus import MilvusClient, DataType
from pymilvus import connections, db
from pymilvus import Collection, CollectionSchema, FieldSchema, utility
from pymilvus import connections
from FlagEmbedding import BGEM3FlagModel
from dotenv import load_dotenv
import os 
import argparse
import pandas as pd
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

system = """너는 회사 핑거 규정들에 대해 잘 알고 있는 전문가야. 핑거 규정에는 다음과 같은 것들이 있어. 
경조금 지급 규정, 화상채팅 가이드, 취업 규칙, 신여비교통비 지급 규칙, 외주인력 출장비 정산, 윤리규정, 취업규칙, 포레스트 통합 메뉴얼, 직무발명보상규정 등
사용자 질문에 대해 최선을 다해 답해줘. 물론, 사용자의 질문에 답하는 튜토리얼을 작성하는 것처럼 대답해 줘
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
qa_no_context = prompt | llm | StrOutputParser()
answer = qa_no_context.invoke(
    {
        "question": "회사 핑거 취업 전 준비사항"
    }
)
print(answer)

from langchain_core.runnables import RunnablePassthrough
hyde_chain = RunnablePassthrough.assign(hypothetical_document=qa_no_context)

answer2 = hyde_chain.invoke(
    {
        "question": "회사 핑거 취업 전 준비사항"
    }
)
print(answer2)