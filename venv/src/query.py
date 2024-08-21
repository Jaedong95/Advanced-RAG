'''
Query Rewrititing Module (Query Decomposition + HyDE)
'''
from abc import ABC, abstractmethod
import os
from src import LLMOpenAI, LLMMistral

class QueryTranslator():
    def __init__(self, args):
        self.args = args
    
    def get_query_status(self, task, prev_query, current_query):
        print(f'{task} 전 쿼리: {prev_query}')
        print(f'{task} 후 쿼리: {current_query}') 
    
    @abstractmethod
    def query_rewrite(self):
        pass 
    
    @abstractmethod
    def query_stepback(self):
        pass 

    @abstractmethod
    def generate_hypothetical_document(self, query, model, tokenizer):
        pass 


class OpenAIQT(QueryTranslator, LLMOpenAI):
    def __init__(self, config):
        QueryTranslator.__init__(self, config)
        LLMOpenAI.__init__(self, config)
    
    def rewrite_query(self, query, model="gpt-4"):
        rewrite_template = f"""주어진 질문에 대해 더 나은 검색 쿼리를 제공해주세요. 검색 쿼리는 '**'로 끝나야 합니다.\n질문: {query} \n개선된 질문:"""
        re_query = self.get_response(rewrite_template, model=model)
        return re_query

    def stepback_query(self, query, model="gpt-4"):
        stepback_template = f""" 당신은 금융권에서 종사하는 전문가입니다.\n당신의 역할은 질문을 보다 일반적인 형태로 바꿔서 좀 더 쉽게 대답할 수 있도록 하는 것입니다.
            몇 가지 예시를 들어보면 다음과 같습니다.\n 
            질문: 회사에 출근할 때 신분증을 가져가야 할까 ? 
            일반적인 형태의 질문: 회사에 출근할 때 필요한게 뭐야 ?

            질문: 국내로 출장갈 때 내가 지원받을 수 있는 금액은 얼마야 ?
            일반적인 형태의 질문: 국내 출장여비 지급 기준에 대해 알려줘

            질문: {query}
            일반적인 형태의 질문:
            """
        stepback_query = self.get_response(stepback_template, model=model)
        return stepback_query


class MistralQT(QueryTranslator, LLMMistral):
    def __init__(self, config):
        QueryTranslator.__init__(self, config)
        LLMMistral.__init__(self, config)

    def rewrite_query(self, query):
        rewrite_template = f"""주어진 질문에 대해 더 나은 검색 쿼리를 제공해주세요. 검색 쿼리는 '**'로 끝나야 합니다.\n질문: {query} \n개선된 질문:"""
        re_query = self.get_response(rewrite_template)
        return re_query

    def stepback_query(self, query):
        stepback_template = f""" 당신은 금융권에서 종사하는 전문가입니다.\n당신의 역할은 질문을 보다 일반적인 형태로 바꿔서 좀 더 쉽게 대답할 수 있도록 하는 것입니다.
            몇 가지 예시를 들어보면 다음과 같습니다.\n 
            질문: 회사에 출근할 때 신분증을 가져가야 할까 ? 
            일반적인 형태의 질문: 회사에 출근할 때 필요한게 뭐야 ?

            질문: 국내로 출장갈 때 내가 지원받을 수 있는 금액은 얼마야 ?
            일반적인 형태의 질문: 국내 출장여비 지급 기준에 대해 알려줘

            위 예시를 참고하여 주어진 query를 간단하고 명료하게 재작성해주세요.

            질문: {query}
            일반적인 형태의 질문:
            """
        stepback_query = self.get_response(stepback_template)
        return stepback_query

class QueryRouter():
    pass