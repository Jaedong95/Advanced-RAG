from dotenv import load_dotenv
from .milvus import MilVus, DataMilVus, MilvusMeta
from .query import OpenAIQT, MistralQT, RulebookQR
from .llm import EmbModel, LLMOpenAI, LLMMistral
import json
import os

class EnvManager():
    def __init__(self, args):
        self.args = args
        load_dotenv()
        self.ip_addr = os.getenv('ip_addr')
        self.cohere_api = os.getenv('COHERE_API_KEY')   
    
    def set_vectordb(self):
        with open(os.path.join(self.args.config_path, self.args.db_config)) as f:
            db_config = json.load(f)
        db_config['ip_addr'] = self.ip_addr

        milvus_db = MilVus(db_config)
        milvus_db.set_env()
        data_milvus = DataMilVus(db_config)
        meta_milvus = MilvusMeta()
        meta_milvus.set_rulebook_map()
        rulebook_eng_to_kor = meta_milvus.rulebook_eng_to_kor

        self.collection = milvus_db.get_collection(self.args.collection_name)
        self.collection.load()

        milvus_db.get_partition(self.collection)
        self.partition_list = [rulebook_eng_to_kor[p_name] for p_name in milvus_db.partition_names if not p_name.startswith('_')]
        return data_milvus

    def set_llm(self):
        with open(os.path.join(self.args.config_path, self.args.llm_config)) as f:
            self.llm_config = json.load(f)
        emb_model = EmbModel()   # Embedding Model: bge-m3
        emb_model.set_embbeding_config()
        response_model = LLMOpenAI(self.llm_config)   # Response Generator
        response_model.set_generation_config()
        return emb_model, response_model 

    def set_query_translator(self):
        llm_qt = OpenAIQT(self.llm_config)   # Query Translator
        llm_qt.set_generation_config()
        return llm_qt

    def set_query_router(self):
        rulebook_qr = RulebookQR()   # Query Router
        rulebook_qr.create_prompt_injection_utterances()
        rulebook_qr.create_rulebook_utterances()
        prompt_injection_route = rulebook_qr.create_route('prompt_injection', rulebook_qr.prompt_injection_utterances)
        rulebook_route = rulebook_qr.create_route('rulebook_check', rulebook_qr.prompt_rulebook_check_utterances)
        route_encoder = rulebook_qr.get_cohere_encoder(self.cohere_api)
        rulebook_rl = rulebook_qr.create_route_layer(route_encoder, [prompt_injection_route, rulebook_route])
        return rulebook_qr, rulebook_rl


class ChatUser():
    def __init__(self, vectordb=None, emb_model=None, response_model=None, query_translator=None, query_router=None, route_layer=None):
        self.vectordb = vectordb
        self.emb_model = emb_model
        self.response_model = response_model 
        self.query_translator = query_translator 
        self.query_router = query_router 
        self.route_layer = route_layer 
    
    def translate_query(self, query):
        rewrite_q = self.query_translator.rewrite_query(query)
        self.query_translator.get_query_status('Query Rewriting', query, rewrite_q)
        stepback_q = self.query_translator.stepback_query(query)
        self.query_translator.get_query_status('Stepback Prompting', rewrite_q, stepback_q)
        return stepback_q

    def retrieve_data(self, query, collection, output_fields='text'):
        cleansed_text = self.vectordb.cleanse_text(query)
        query_emb = self.emb_model.bge_embed_data(cleansed_text)
        self.vectordb.set_search_params(query_emb, output_fields=output_fields)   
        search_params = self.vectordb.search_params
        self.search_result = self.vectordb.search_data(collection, search_params)

    def postprocess_data(self, threshold):
        id_list = []; dist_list = []
        '''
        if reranking is needed
        '''
        retreived_txt = self.vectordb.decode_search_result(self.search_result)
        id_list, dist_list = self.vectordb.get_distance(self.search_result)
        threshold_txt = self.vectordb.check_l2_threshold(retreived_txt, threshold, dist_list[0])
        return threshold_txt

    def continue_conv(self, flag):
        continue_conv = input('계속 대화하시겠습니까 ? (y/n): ')
        if continue_conv.lower() == 'y':
            flag = True 
            print(f'대화를 계속합니다.')
        else:
            flag = False
            print(f'대화를 종료합니다.')

        return flag 