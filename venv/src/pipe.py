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
        llm_rs = LLMOpenAI(self.llm_config)   # Response Generator
        llm_rs.set_generation_config()
        return emb_model, llm_rs 

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
    def __init__(self, args):
        self.args = args 
    
    pass