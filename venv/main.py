from dotenv import load_dotenv
from src import MilVus, DataMilVus, MilvusMeta
from src import OpenAIQT, MistralQT, RulebookQR
from src import EmbModel, LLMOpenAI, LLMMistral
from src import EnvManager, ChatUser
import argparse
import json
import os 

def main(args):
    load_dotenv() 
    ip_addr = os.getenv('ip_addr')
    cohere_api = os.getenv('COHERE_API_KEY')
    if not ip_addr or not cohere_api:
        raise ValueError("IP 주소나 Cohere API 키가 제대로 로드되지 않았습니다.")
    
    env_manager = EnvManager(args)
    data_milvus = env_manager.set_vectordb()
    emb_model, response_model = env_manager.set_llm()
    query_translator = env_manager.set_query_translator()
    query_router, route_layer = env_manager.set_query_router()
    rulebook_chatbot = ChatUser(vectordb=data_milvus, emb_model=emb_model, response_model=response_model, 
                    query_translator=query_translator, query_router=query_router, route_layer=route_layer)

    flag = True 
    threshold = 0.65
    
    print(f"대화를 시작해보세요 ! 회사 내부 규정에 대해 설명해주는 챗봇입니다.")
    print(f"* 현재 {', '.join(env_manager.partition_list)}에 대한 질의응답이 가능합니다.")
    while(flag):
        query = input('사용자: ')
        print(f'Step 1. Query를 재조정합니다.')
        stepback_q = rulebook_chatbot.translate_query(query)

        print(f'Step 2. Query를 Routing합니다.')
        routed_query = rulebook_chatbot.query_router.semantic_layer(route_layer, stepback_q)
        print(f'routed query: {routed_query}', end='\n\n')
        if rulebook_chatbot.route_layer(stepback_q).name == 'prompt_injection':    # 부적절한 발화가 입력된 경우   
            print(rulebook_chatbot.response_model.get_response(routed_query))
        else:
            print(f'Step 3. 가상 문서를 생성합니다.')
            hyde = rulebook_chatbot.response_model.get_response(stepback_q)
            print(f'생성된 문서: {hyde}', end='\n\n')
            
            print(f'Step 4. 관련 정보를 추출합니다.')
            rulebook_chatbot.retrieve_data(hyde, env_manager.collection)
            # print(f'추출된 정보: {search_result}', end='\n\n')
            
            print(f'Step 5. 후속 처리(Re-ranking, Check distance)를 진행합니다.')
            threshold_txt = rulebook_chatbot.postprocess_data(threshold)
            print(f'후처리된 정보: {threshold_txt}', end='\n\n')

            print(f'Step 6. 응답을 생성합니다.') 
            prompt_template = rulebook_chatbot.response_model.set_prompt_template(routed_query, threshold_txt)
            response = rulebook_chatbot.response_model.get_response(prompt_template)
            print(f'챗봇: {response}')
        
        flag = rulebook_chatbot.continue_conv(flag)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='config/')
    cli_parser.add_argument('--db_config', type=str, default='db_config.json')
    cli_parser.add_argument('--llm_config', type=str, default='llm_config.json')
    cli_parser.add_argument('--output_dir', type=str, default='../../data/pdf/embed_output')
    cli_parser.add_argument('--collection_name', type=str, default='rule_book')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)