from dotenv import load_dotenv
from src import MilVus, DataMilVus, MilvusMeta
from src import OpenAIQT, MistralQT, RulebookQR
from src import EmbModel, LLMOpenAI, LLMMistral
from src import EnvManager, ChatUser
import argparse
import logging
import json
import os 

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),# mode='w'),  # 로그를 파일에 기록
        # logging.StreamHandler()  # 콘솔에도 출력
    ]
)

logging.basicConfig(filename='warnings.log', level=logging.WARNING)
logging.captureWarnings(True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    load_dotenv() 
    ip_addr = os.getenv('ip_addr')
    cohere_api = os.getenv('COHERE_API_KEY')
    if not ip_addr or not cohere_api:
        raise ValueError("IP 주소나 Cohere API 키가 제대로 로드되지 않았습니다.")
    
    # 로거 생성
    logger = logging.getLogger(__name__)   
    env_manager = EnvManager(args)
    data_milvus = env_manager.set_vectordb()
    emb_model, response_model = env_manager.set_llm()

    use_query_transform = args.use_query_transform
    use_query_routing = args.use_query_routing

    env_manager.set_query_transformer(use_query_transform)
    env_manager.set_query_router(use_query_routing)

    rulebook_chatbot = ChatUser(vectordb=data_milvus, emb_model=emb_model, response_model=response_model, 
                    query_transformer=env_manager.query_transformer, query_router=env_manager.query_router, route_layer=env_manager.route_layer, logger=logger)

    conv_flag = True; injection_flag = False
    threshold = 0.65
    
    print(f"대화를 시작해보세요 ! 회사 내부 규정에 대해 설명해주는 챗봇입니다.")
    print(f"* 현재 {', '.join(env_manager.partition_list)}에 대한 질의응답이 가능합니다.")
    while conv_flag:
        query = input('사용자: ')
        if use_query_transform:    
            print(f'Step 1. Query를 재조정합니다.')
            query = rulebook_chatbot.transform_query(query) 
        prompt_query = query

        if use_query_routing:
            print(f'Step 2. Query를 Routing합니다.')
            query, injection_flag = rulebook_chatbot.route_query(query)
            if injection_flag == True:   # Prompt Injection이 감지된 경우 대화 종료
                logger.info(f'Prompt Injection이 감지되었습니다 !')
                print(query)
                conv_flag = rulebook_chatbot.continue_conv()
                continue
        
        print(f'Step 3. 가상 문서를 생성합니다.')
        hyde = rulebook_chatbot.response_model.get_response(query)
        logger.info(f'생성된 문서: {hyde}')
        
        print(f'Step 4. 관련 정보를 추출합니다.')
        rulebook_chatbot.retrieve_data(hyde, env_manager.collection)
        logger.info(f'추출된 정보: {rulebook_chatbot.search_result}')
        
        print(f'Step 5. 후속 처리(Re-ranking, Check distance)를 진행합니다.')
        threshold_txt = rulebook_chatbot.postprocess_data(threshold)
        logger.info(f'후처리된 정보: {threshold_txt}')

        print(f'Step 6. 응답을 생성합니다.') 
        prompt_template = rulebook_chatbot.response_model.set_prompt_template(prompt_query, threshold_txt)
        response = rulebook_chatbot.response_model.get_response(prompt_template)
        print(f'챗봇: {response}')
        conv_flag = rulebook_chatbot.continue_conv()


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='config/')
    cli_parser.add_argument('--db_config', type=str, default='db_config.json')
    cli_parser.add_argument('--llm_config', type=str, default='llm_config.json')
    cli_parser.add_argument('--output_dir', type=str, default='../../data/pdf/embed_output')
    cli_parser.add_argument('--collection_name', type=str, default='rule_book')
    cli_parser.add_argument('--use_query_transform', type=str2bool, default=True)
    cli_parser.add_argument('--use_query_routing', type=str2bool, default=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)