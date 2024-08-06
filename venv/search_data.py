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

def cleanse_text(text):
    '''
    다중 줄바꿈 제거 및 특수 문자 중복 제거
    '''
    import re 
    text = re.sub(r'(\n\s*)+\n+', '\n\n', text)
    text = re.sub(r"\·{1,}", " ", text)
    text = re.sub(r"\.{1,}", ".", text)
    # print('after cleansing: ' + text)
    return text

def main(args):
    load_dotenv()
    db_name = 'finger'
    ip_addr = os.getenv('ip_addr')

    conn = connections.connect(
        alias="default", 
        host=ip_addr, 
        port='19530'
    )
    print(f'conn: {conn}')

    collection_name = "fin_collection" 
    collection = Collection(collection_name)
    collection.load()
    print(Collection(collection_name))
    print(db.list_database())
    
    # User Query 
    text = """취업규칙 제 1장 총칙에 관해서 """
    cleansed_text = cleanse_text(text)
    model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)
    embeddings_1 = model.encode(cleansed_text, 
                            batch_size=12, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
    embeddings_1 = list(map(np.float32, embeddings_1))
    print(embeddings_1[:3])

    search_params = {
    "metric_type": "L2", 
    "offset": 5, 
    "ignore_growing": False, 
    "params": {"nprobe": 5}
    }
    results = collection.search(
        data=[embeddings_1], 
        anns_field="text", 
        # the sum of `offset` in `param` and `limit` 
        # should be less than 16384.
        param=search_params,
        limit=5,
        expr=None,
        # set the names of the fields you want to 
        # retrieve from the search result.
        output_fields=['data_id'],
        consistency_level="Strong"
    )
    print(results)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--document_path', type=str, default='../data/pdf')
    cli_parser.add_argument('--file_name', type=str, default='취업규칙.csv')
    cli_parser.add_argument('--output_dir', type=str, default='../data/output')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)