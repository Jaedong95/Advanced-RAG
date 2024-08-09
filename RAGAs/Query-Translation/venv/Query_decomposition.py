import os
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
openai_api = os.getenv('OPENAI_API_KEY')

# Query Rewrite 
def get_rewrite_query(query):
    rewrite_template = f""" 주어진 질문에 대해 더 나은 검색 쿼리를 제공해주세요. 검색 쿼리는 '**'로 끝나야 합니다.\n질문: {query} \n개선된 질문:"""
    return rewrite_template

# Stepback-Prompting 
def get_stepback_query(query):
    stepback_template = f""" 당신은 금융권에서 종사하는 전문가입니다.\n당신의 역할은 질문을 보다 일반적인 형태로 바꿔서 좀 더 쉽게 대답할 수 있도록 하는 것입니다.
    몇 가지 예시를 들어보면 다음과 같습니다.\n 
    질문: 회사에 출근할 때 신분증을 가져가야 할까 ? 
    일반적인 형태의 질문: 회사에 출근할 때 필요한게 뭐야 ?

    질문: 국내로 출장갈 때 내가 지원받을 수 있는 금액은 얼마야 ?
    일반적인 형태의 질문: 국내 출장여비 지급 기준에 대해 알려줘

    질문: {query}
    일반적인 형태의 질문:"""
    return stepback_template

def gen(x):
    generation_config = GenerationConfig(
        temperature=0.8,
        top_p=0.8,
        top_k=100,
        max_new_tokens=1024,
        early_stopping=True,
        do_sample=True,
    )
    q = f"[INST]{x} [/INST]"
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # streamer=streamer,
    )
    result_str = tokenizer.decode(gened[0])
    start_tag = f"\n\n### Response: "
    start_index = result_str.find(start_tag)

    if start_index != -1:
        result_str = result_str[start_index + len(start_tag):].strip()
    return result_str

query = "회사에 입사할 때 제출해야 하는게 뭐야 ?"
rewrite_query = get_rewrite_query(query)
stepback_query = get_stepback_query(query)
# print(f'rewrite_query: {rewrite_query}')
# print(f'stepback_query: {stepback_query}')

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer, GenerationConfig
import openai

'''
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
'''

model_name='davidkim205/komt-mistral-7b-v1'
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f'step-back query of open source model:{gen(stepback_query)}')

response = openai.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": stepback_query}])
print(f'step-back query of OpenAI:{response.choices[0].message}')