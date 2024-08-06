'''
전처리한 데이터 파일을 읽어와 VectorDB에 임베딩한다. 
사용 벡터 DB: Milvus
대상 Collection: Finger-rulebook
임베딩 모델: bge-M3 Model
'''
from pymilvus import MilvusClient, DataType
from pymilvus import connections, db
from pymilvus import Collection, CollectionSchema, FieldSchema, utility
from pymilvus import connections
from dotenv import load_dotenv
import os 
import argparse
import pandas as pd
import numpy as np

def collection_info(collection_name):
    print(f'collection info')
    collection = Collection(collection_name)
    print(f'schema info: {collection.schema}') 
    print(f'collection info: {collection.description}')
    print(f'collection name: {collection.name}')
    print(f'is collection empty ?: {collection.is_empty}')
    print(f'num of data: {collection.num_entities}')
    print(f'primary key of collection: {collection.primary_field}')
    print(f'partition of collection: {collection.partition}')

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

    try:
        assert db.create_database(db_name)
    except:
        pass 

    collection_name = "fin_collection" 
    print(Collection(collection_name))
    print(db.list_database())
    print(collection_info(collection_name))

    # Load Data 
    data_df = pd.read_csv(os.path.join(args.output_dir, args.file_name), index_col=0)
    
    # Embed Data  
    text = data_df['text'].values
    
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)
    embeddings_1 = model.encode(list(text), 
                            batch_size=12, 
                            max_length=8192,   # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
    embeddings_1 = list(map(np.float32, embeddings_1))
    print(np.shape(embeddings_1))

    # Insert data 
    data_id = [id for id in range(len(text))]
    text_data = embeddings_1
    source = list(data_df['source'].values)
    page_no = list(data_df['page_no'].values)
    chunk_size = list(data_df['chunk_size'].values)

    data = [
        data_id,
        text_data,
        source,
        page_no,
        chunk_size
    ]
    collection = Collection(collection_name)
    collection.insert(data, partition_name="employment_rules")

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--document_path', type=str, default='../data/pdf')
    cli_parser.add_argument('--file_name', type=str, default='취업규칙.csv')
    cli_parser.add_argument('--output_dir', type=str, default='../data/output')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)