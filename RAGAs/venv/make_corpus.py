import os 
import argparse 
from autorag.utils import cast_corpus_dataset 
from llama_index.core import SimpleDirectoryReader 
from llama_index.core.node_parser import TokenTextSplitter 
from autorag.data.corpus import llama_text_node_to_parquet 
from dotenv import load_dotenv

load_dotenv()
openai_api = os.getenv('OPENAI_API_KEY')

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
    documents = SimpleDirectoryReader(input_files=[os.path.join(args.document_path, args.file_name)]).load_data()
    nodes = TokenTextSplitter().get_nodes_from_documents(documents=documents, chunk_size=128, chunk_overlagp=32)
    corpus_df = llama_text_node_to_parquet(nodes)
    corpus_df = cast_corpus_dataset(corpus_df)
    corpus_df['contents'] = corpus_df['contents'].apply(cleanse_text)
    corpus_df.to_csv(os.path.join(args.output_dir, args.file_name.split('.')[0] + '.csv'), index=False)

    import pandas as pd 
    corpus_df = pd.read_csv(os.path.join(args.output_dir, args.file_name.split('.')[0] + '.csv'), engine='pyarrow')
    print(corpus_df.head(3))

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--document_path', type=str, default='../data/pdf')
    cli_parser.add_argument('--file_name', type=str, default='경조금지급규정.pdf')
    cli_parser.add_argument('--output_dir', type=str, default='../data/pdf/output')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)