from dotenv import load_dotenv
from src import MilVus, DataMilVus
from src import OpenAIQT, MistralQT, RulebookQR
from src import EmbModel, LLMOpenAI, LLMMistral
import os 
import argparse
import json

def main(args):
    load_dotenv()
    db_name = 'finger'
    ip_addr = os.getenv('ip_addr')
    cohere_api = os.getenv('COHERE_API_KEY')

    with open(os.path.join(args.config_path, args.db_config)) as f:
        db_config = json.load(f)
    db_config['ip_addr'] = ip_addr

    with open(os.path.join(args.config_path, args.llm_config)) as f:
        llm_config = json.load(f)

    milvus_db = MilVus(db_config)
    milvus_db.set_env()
    data_milvus = DataMilVus(db_config)
    print(f'client: {milvus_db.client}')
    
    collection = milvus_db.get_collection(args.collection_name)
    collection.load()
    
    emb_model = EmbModel()   # Embedding Model: bge-m3
    emb_model.set_embbeding_config()
    
    llm_qt = OpenAIQT(llm_config)   # Query Translator
    llm_qt.set_generation_config()

    rulebook_qr = RulebookQR()   # Query Router
    rulebook_qr.create_prompt_injection_utterances()
    rulebook_qr.create_rulebook_utterances()
    prompt_injection_route = rulebook_qr.create_route('prompt_injection', rulebook_qr.prompt_injection_utterances)
    rulebook_route = rulebook_qr.create_route('rulebook_check', rulebook_qr.prompt_rulebook_check_utterances)
    route_encoder = rulebook_qr.get_cohere_encoder(cohere_api)
    rulebook_rl = rulebook_qr.create_route_layer(route_encoder, [prompt_injection_route, rulebook_route])

    llm_rs = LLMOpenAI(llm_config)   # Response Generator
    llm_rs.set_generation_config()

    flag = True 
    threshold = 0.6
    print('대화를 시작해보세요 ! 회사 내부 규정에 대해 설명해주는 챗봇입니다.')
    while(flag):
        query = input('사용자: ')
        print(f'Step 1. Query를 재조정합니다.')
        rewrite_q = llm_qt.rewrite_query(query)
        llm_qt.get_query_status('Query Rewriting', query, rewrite_q)
        stepback_q = llm_qt.stepback_query(query)
        llm_qt.get_query_status('Stepback Prompting', rewrite_q, stepback_q)

        # Query Routing 
        routed_query = rulebook_qr.semantic_layer(rulebook_rl, stepback_q)
        llm_qt.get_query_status('Query Routing', stepback_q, routed_query)

        print('')
        print(f'Step 2. 가상 문서를 생성합니다.')
        hyde = llm_qt.get_response(stepback_q)
        print(f'생성된 문서: {hyde}', end='\n\n')
        
        print(f'Step 3. 관련 정보를 추출합니다.')
        cleansed_text = data_milvus.cleanse_text(hyde)
        query_emb = emb_model.bge_embed_data(cleansed_text)
        data_milvus.set_search_params()
        search_result = data_milvus.search_data(collection, query_emb, output_fields='text')
        
        print('')
        print(f'Step 4. 추출된 정보를 재조정합니다.')
        id_list = []; dist_list = []
        retreived_txt = data_milvus.decode_search_result(search_result)
        id_list, dist_list = data_milvus.get_distance(search_result)
        if dist_list[0] > threshold:
            print(f'Euclidean distance: {dist_list[0]}')
            retreived_txt = "모르는 정보입니다." 
        print(f'추출된 정보: {retreived_txt}', end='\n\n')
        
        # Augment data 
        prompt_template = llm_rs.set_prompt_template(stepback_q, retreived_txt)

        # Generate data
        print(f'Step 5. 응답을 생성합니다.') 
        response = llm_rs.get_response(prompt_template)
        print(f'챗봇: {response}')
        print(f'응답 결과를 평가합니다 .. * Evaluation metric: ragas')
        
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
    cli_parser.add_argument('--file_name', type=str, default='취업규칙.csv')
    cli_parser.add_argument('--collection_name', type=str, default='rule_book')
    # cli_parser.add_argument('--partition_name', type=str, default=None)
    cli_argse = cli_parser.parse_args()
    main(cli_argse)