from pymilvus import connections, db
from pymilvus import Collection, CollectionSchema, FieldSchema, utility, MilvusClient, DataType
from dotenv import load_dotenv
import os

load_dotenv()
db_name = 'finger'
ip_addr = os.getenv('ip_addr')

conn = connections.connect(
  alias="default", 
  host='192.168.0.146', 
  port='19530',
  # db_name=None, 
)
print(conn)

# 1. Set up a Milvus client
client = MilvusClient(
    uri="http://" + ip_addr + ":19530", port=19530
)
collection_name = "fin_collection" 
# Step 1. Create Database 
try:
    assert db.create_database(db_name)
except:
    pass 
# assert utility.has_collection(collection_name)
try:
    assert utility.has_collection(collection_name)
    utility.drop_collection(collection_name)
except:
    print('error !')
    pass 

# Step 2. Create collection 
## 2.1 Create schema
data_id = FieldSchema(
  name="data_id",
  dtype=DataType.INT64,
  is_primary=True,
)
data_text = FieldSchema(
    name="text",
    dtype=DataType.FLOAT_VECTOR,
    is_primary=False,
    dim=1024
)
data_source = FieldSchema(
    name="source",
    dtype=DataType.VARCHAR,
    max_length=200
)
data_pageno = FieldSchema(
    name="page_no",
    dtype=DataType.INT64,
    is_primary=False
)
data_chunksize = FieldSchema(
    name="chunk_size",
    dtype=DataType.INT64,
    is_primary=False
)
schema = CollectionSchema(
  fields=[data_id, data_text, data_source, data_pageno, data_chunksize],
  description="Test data search",
  enable_dynamic_field=True
)

## 2.2 Create collection
collection_name = "fin_collection" 
collection = Collection(
    name=collection_name,
    schema=schema,
    using='default',
    shards_num=2
)
print(collection)

## 2.3 Build index 
index_params = {
  "metric_type": "L2",    # type of metrics used to measure the similarity of vectors 
  "index_type": "IVF_FLAT",   # type of index used to accelerate vector search 
  "params": {"nlist": 65536},
}

collection.create_index(
    field_name = "text",
    index_params = index_params
)
print(f'building index: {utility.index_building_progress(collection_name)}')

## 2.4 Create Partition
## 윤리규정, 복지제도, ... 파일별 파티션
try:
    assert collection.create_partition("code_of_ethics")
    assert collection.create_partition("employment_rules")
    # assert collection.create_partition("")
except:
    pass

print(f'partitions: {collection.partitions}')