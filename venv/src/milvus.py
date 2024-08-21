from pymilvus import MilvusClient, DataType
from pymilvus import connections, db
from pymilvus import Collection, CollectionSchema, FieldSchema, utility
from pymilvus import connections
from .data_p import DataProcessor
import numpy as np

class MilVus():
    def __init__(self, args):
        self.ip_addr = args['ip_addr'] 
        self.port = '19530'

    def set_env(self):
        self.client = MilvusClient(
            uri="http://" + self.ip_addr + ":19530", port=19530
        )
        self.conn = connections.connect(
            alias="default", 
            host=self.ip_addr, 
            port='19530'
        )
    
    def create_db(self, db_name):
        try:
            assert db.create_database(db_name), f'{db_name}은 이미 존재합니다.'
            # db.create_database(db_name)
        except:
            print(f'보유 Database: {db.list_database()}')
            pass

    def create_collection(self, collection_name, schema, shards_num):
        collection = Collection(
            name=collection_name,
            schema=schema,
            using='default',
            shards_num=shards_num
        )
        return collection 

    def create_field_schema(self, schema_name, dtype=None, dim=1024, max_length=200, is_primary=False):
        data_type = None
        if dtype == 'int':
            data_type = DataType.INT64 
        elif dtype == 'str':
            data_type = DataType.VARCHAR
        elif dtype == 'float':   # vector
            data_type = DataType.FLOAT_VECTOR
        field_schema = FieldSchema(
            name=schema_name,
            dtype=data_type,
            is_primary=is_primary,
            dim=dim, 
            max_length=max_length
        )
        return field_schema

    def create_schema(self, field_schema_list, desc, enable_dynamic_field=True):
        schema = CollectionSchema(
            fields=field_schema_list,
            description=desc,
            enable_dynamic_field=enable_dynamic_field
        )
        print(f'created Schema !')
        return schema

    def create_index(self, collection, field_name):
        index_params = {
            "metric_type": "L2",    # type of metrics used to measure the similarity of vectors 
            "index_type": "IVF_FLAT",   # type of index used to accelerate vector search 
            "params": {"nlist": 65536},
        }   
        collection.create_index(
            field_name = field_name,
            index_params = index_params
        )
        # print(f'building index on: {utility.index_building_progress(field_name)}')
    
    def create_partition(self, collection, partition_name):
        try:
            assert collection.create_partition(partition_name), f'{partition_name}은 이미 존재하는 파티션입니다.'
        except:
            pass
        print(f'partitions: {collection.partitions}')

    def delete_collection(self, collection_name):
        try:
            assert utility.has_collection(collection_name), f'{collection_name}이 존재하지 않습니다.'
            utility.drop_collection(collection_name)
        except:
            pass

    def get_collection(self, collection_name):
        collection = Collection(collection_name)
        return collection

    def get_partition(self, collection):
        partitions = collection.partitions 
        for partition in partitions: 
            print(f'partition name: {partition.name} num of entitiy: {partition.num_entities}')
    

    def collection_info(self, collection_name):
        print(f'collection info')
        collection = Collection(collection_name)
        print(f'schema info: {collection.schema}') 
        print(f'collection info: {collection.description}')
        print(f'collection name: {collection.name}')
        print(f'is collection empty ?: {collection.is_empty}')
        print(f'num of data: {collection.num_entities}')
        print(f'primary key of collection: {collection.primary_field}')
        print(f'partition of collection: {collection.partition}')


class DataMilVus(DataProcessor):   #  args: (DataProcessor)
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def set_env(self):
        self.client = MilvusClient(
            uri="http://" + self.args.ip_addr + ":19530", port=19530
        )
        self.conn = connections.connect(
            alias="default", 
            host=self.args.ip_addr, 
            port='19530'
        )

    def bge_milvus_embed(self, text):   
        ''' 2.4 x ''' 
        from pymilvus.model.hybrid import BGEM3EmbedddingFunction
        bge_m3_ef = BGEM3EmbedddingFunction(
            model_name='BAAI/bge-m3',
            device='cpu',
            use_fp16=False
        )
        
        if isinstance(text, str):
            bge_emb = bge_m3_ef.encode_queries(text)
            print(f"embeddings (dense): {bge_emb['dense']}")
        else:
            bge_emb = bge_m3_ef.encode_documents(text)
        return bge_emb

    def insert_data(self, m_data, collection_name, partition_name=None):
        collection = Collection(collection_name)
        collection.insert(m_data, partition_name)
        
    def get_len_data(self, collection):
        print(collection.num_entities)

    def set_search_params(self, metric_type="L2", offset=5, ignore_growing=False):
        self.search_params = {
            "metric_type": metric_type, 
            # "offset": offset, 
            # "ignore_growing": ignore_growing, 
            # "params": {"nprobe": 80} 
        }
    
    def search_data(self, collection, query_emb, limit=5, anns_field='text_emb', output_fields=None, consistency_level="Strong"):
        results = collection.search(
                data=[query_emb], 
                anns_field=anns_field, 
                # the sum of `offset` in `param` and `limit` should be less than 16384.
                param=self.search_params,
                limit=5,
                expr=None,
                # set the names of the fields you want to retrieve from the search result.
                output_fields=[output_fields],
                consistency_level=consistency_level
            )
        return results

    def get_distance(self, search_result):
        for idx in range(len(search_result[0])):
            print(search_result[0][idx].id, search_result[0][idx].distance)

    def decode_search_result(self, search_result):
        # print(f'ids: {search_result[0][0].id}')
        print(f"entity: {search_result[0][0].entity.get('text')}") 
        return search_result[0][0].entity.get('text')