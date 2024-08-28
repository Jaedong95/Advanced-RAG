from dotenv import load_dotenv
from src import MilVus, DataMilVus, MilvusMeta
from src import OpenAIQT, MistralQT, RulebookQR
from src import EmbModel, LLMOpenAI, LLMMistral
from src import ProjectManager
import os 
import argparse
import json




def main(args):
    load_dotenv() 
    ip_addr = os.getenv('ip_addr')
    cohere_api = os.getenv('COHERE_API_KEY')

    data_milvus, partition_list, collection = set_vectordb(args, ip_addr)
    emb_model, llm_rs = set_llm(args)
    llm_qt = set_query_translator(llm_config)
    rulebook_qr, rulebook_rl = set_query_router(cohere_api)
    
    flag = True 
    threshold = 0.65
    
    print(f"대화를 시작해보세요 ! 회사 내부 규정에 대해 설명해주는 챗봇입니다.")
    print(f"* 현재 {', '.join(partition_list)}에 대한 질의응답이 가능합니다.")
    while(flag):
        query = input('사용자: ')
        print(f'Step 1. Query를 재조정합니다.')
        rewrite_q = llm_qt.rewrite_query(query)
        llm_qt.get_query_status('Query Rewriting', query, rewrite_q)
        stepback_q = llm_qt.stepback_query(query)
        llm_qt.get_query_status('Stepback Prompting', rewrite_q, stepback_q)

        print(f'Step 2. Query를 Routing합니다.')
        routed_query = rulebook_qr.semantic_layer(rulebook_rl, stepback_q)
        print(f'routed query: {routed_query}', end='\n\n')
        if rulebook_rl(stepback_q).name == 'prompt_injection':    # 부적절한 발화가 입력된 경우   
            print(llm_rs.get_response(routed_query))
        else:
            print(f'Step 3. 가상 문서를 생성합니다.')
            hyde = llm_qt.get_response(stepback_q)
            print(f'생성된 문서: {hyde}', end='\n\n')
            
            print(f'Step 4. 관련 정보를 추출합니다.')
            cleansed_text = data_milvus.cleanse_text(hyde)
            query_emb = emb_model.bge_embed_data(cleansed_text)
            data_milvus.set_search_params(query_emb, output_fields='text')   
            search_params = data_milvus.search_params
            search_result = data_milvus.search_data(collection, search_params)
            # print(f'추출된 정보: {search_result}', end='\n\n')
            
            print(f'Step 5. 추출된 정보를 재조정합니다.')
            id_list = []; dist_list = []
            retreived_txt = data_milvus.decode_search_result(search_result)
            id_list, dist_list = data_milvus.get_distance(search_result)
            threshold_txt = data_milvus.check_l2_threshold(retreived_txt, threshold, dist_list[0])
            print(f'후처리된 정보: {threshold_txt}', end='\n\n')
            
            # Augment data 
            prompt_template = llm_rs.set_prompt_template(routed_query, threshold_txt)

            # Generate data
            print(f'Step 6. 응답을 생성합니다.') 
            response = llm_rs.get_response(prompt_template)
            print(f'챗봇: {response}')
        
        continue_conv = input('계속 대화하시겠습니까 ? (y/n): ')
        if continue_conv == 'y':
            flag = True 
            print(f'대화를 계속합니다.')
        else:
            flag = False
            print(f'대화를 종료합니다.')

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='config/')
    cli_parser.add_argument('--db_config', type=str, default='db_config.json')
    cli_parser.add_argument('--llm_config', type=str, default='llm_config.json')
    cli_parser.add_argument('--output_dir', type=str, default='../../data/pdf/embed_output')
    cli_parser.add_argument('--collection_name', type=str, default='rule_book')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)