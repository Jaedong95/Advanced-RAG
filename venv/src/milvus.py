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
        self.partitions = collection.partitions 
        self.partition_names = [] 
        self.partition_entities_num = [] 
        for partition in self.partitions: 
            print(f'partition name: {partition.name} num of entitiy: {partition.num_entities}')
            self.partition_names.append(partition.name)
            self.partition_entities_num.append(partition.num_entities)

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
    '''
    구축된 Milvus DB에 대한 data search, insert 등 작업 수행
    '''
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

    def insert_data(self, m_data, collection_name, partition_name=None):
        collection = Collection(collection_name)
        collection.insert(m_data, partition_name)
        
    def get_len_data(self, collection):
        print(collection.num_entities)

    def set_search_params(self, query_emb, anns_field='text_emb', metric_type="L2", expr=None, limit=5, output_fields=None, consistency_level="Strong"):
        self.search_params = {
            "data": [query_emb],
            "anns_field": anns_field, 
            "param": {"metric_type": metric_type, "params": {"nprobe": 0}, "offset": 0},
            "limit": limit,
            "expr": expr, 
            "output_fields": [output_fields],
            "consistency_level": consistency_level
        }
    
    def search_data(self, collection, search_params):
        results = collection.search(**search_params)
        return results

    def get_distance(self, search_result):
        id_list = [] 
        distance_list = [] 
        for idx in range(len(search_result[0])):
            id_list.append(search_result[0][idx].id)
            distance_list.append(search_result[0][idx].distance)
        return id_list, distance_list

    def decode_search_result(self, search_result):
        # print(f'ids: {search_result[0][0].id}')
        print(f"entity: {search_result[0][0].entity.get('text')}") 
        return search_result[0][0].entity.get('text')

    def rerank_data(self, search_result):
        pass 


class MilvusMeta(MilVus):
    ''' 
    파일이름 - ID Code, 파일이름 - 영문이름 (파티션) 매핑 정보 관리 클래스 
    '''
    def set_rulebook_map(self):
        self.rulebook_id_code = {
            '취업규칙': '00', 
            '윤리규정': '01', 
            '신여비교통비': '02', 
            '경조금지급규정': '03'
        }
        self.rulebook_kor_to_eng = {
            '취업규칙': 'employment_rules',
            '윤리규정': 'code_of_ethics',
            '신여비교통비': 'transport_expenses',
            '경조금지급규정': 'extra_expenditure',
        }
