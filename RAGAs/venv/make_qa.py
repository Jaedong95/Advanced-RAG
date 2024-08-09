import os 
import argparse
import pandas as pd
from autorag.utils import cast_corpus_dataset
from llama_index.llms.openai import OpenAI
from autorag.data.qacreation import generate_qa_llama_index, make_single_content_qa
from llama_index.core import SimpleDirectoryReader 
from llama_index.core.node_parser import TokenTextSplitter 
from autorag.data.corpus import llama_text_node_to_parquet 
from dotenv import load_dotenv

load_dotenv()
openai_api = os.getenv('OPENAI_API_KEY')

def main(args):
    prompt = """다음은 자사의 {{args.file_name.split('.')[0]}}에 관한 내용입니다. 
    해당 파일을 보고 할 만한 질문을 만드세요.
    반드시 해당 파일에 관련한 질문이어야 합니다. 
    
    규정: 
    {{text}}

    생성할 질문 개수: {{num_questions}}

    예시
    [Q]: 윤리강령 중 임직원의 기본 윤리 제 7조의 내용은 ?
    [A]: 제 7조 (정보보호)
    1 고객의 정보와 자산의 반, 출입은 업무 목적으로만 가능하며, 회사의 반, 출입 절차를 준수한다. 
    2 직무 수행 중 취득한 정보는 재직 중이나 퇴직 후에도 사전허가나 승인 없이 절대로 유출하지 않는다. 
    3 업무활동상의 각종 정보는 사실에 근거하여 정확하고 명확하게 기록하고, 잘 관리한다. 

    결과:
    """
    corpus_df = pd.read_csv(os.path.join(args.output_dir, args.file_name.split('.')[0] + '.csv'), engine='pyarrow')
    
    llm = OpenAI(model='gpt-4o', temperature=0.7)
    qa_df = make_single_content_qa(corpus_df, content_size=args.qa_size, qa_creation_func=generate_qa_llama_index, llm=llm, prompt=prompt, question_num_per_content=1)
    qa_df = qa_df.loc[~qa_df['query'].str.contains('회사 규정에 관한 내용이 아닙니다.')]
    qa_df.reset_index(drop=True, inplace=True)
    qa_df.to_csv(os.path.join(args.output_dir, 'qa_' + args.file_name.split('.')[0] + '.csv'), index=False)
    new_corpus_df = pd.read_csv(os.path.join(args.output_dir, 'qa_' + args.file_name.split('.')[0] + '.csv'), engine='pyarrow')
    # print(new_corpus_df.head(3))

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--document_path', type=str, default='../data/pdf')
    cli_parser.add_argument('--file_name', type=str, default='경조금지급규정.csv')
    cli_parser.add_argument('--qa_size', type=int, default=10)
    cli_parser.add_argument('--output_dir', type=str, default='../data/pdf/output')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)