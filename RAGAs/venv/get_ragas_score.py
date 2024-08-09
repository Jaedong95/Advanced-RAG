import pandas as pd 
import os 
import torch
import argparse
import ragas
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import TextStreamer, GenerationConfig
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy, 
    faithfulness, 
    context_recall, 
    context_precision
)

load_dotenv()
openai_api = os.getenv('OPENAI_API_KEY')

def main(args):
    corpus_df = pd.read_csv(os.path.join(args.output_dir, 'response_' + args.file_name.split('.')[0] + '.csv'), engine='pyarrow')
    '''
    origin_df = pd.read_csv(os.path.join(args.output_dir, args.file_name.split('.')[0] + '.csv'), engine='pyarrow')
    print(origin_df.columns)
    print(corpus_df.columns)  
    print(len(origin_df), len(corpus_df))'''  
    
    ragas_df = corpus_df[['query', 'response', 'generation_gt']]
    # print(str(ragas_df.generation_gt[0].split("'")[1]))
    ragas_df.columns = ['question', 'ground_truth', 'contexts']
    ragas_df['contexts'] = ragas_df['contexts'].apply(lambda x: x.split("'")[1])
    ragas_df['contexts'] = ragas_df['contexts'].apply(lambda x: [x])
    print(ragas_df.contexts[0])
    ragas_df['answer'] = ragas_df['ground_truth']
    ragas_df['question_type'] = 'simple'
    ragas_df['episode_done'] = True
    # print(ragas_df.head())
    ragas_df.to_csv(os.path.join(args.output_dir, 'ragas_' + args.file_name.split('.')[0] + '.csv'), index=False)
    # corpus_df = corpus_df[['retrieval_gt', 'qid', 'query', 'generation']]
    # corpus_df.columns = ['retrieval_gt', 'qid', 'question', 'context', 'answer', ]
    dataset = Dataset.from_pandas(ragas_df)
    result = evaluate(
        dataset, 
        metrics=[
            context_precision, 
            faithfulness, 
            answer_relevancy,
            context_recall
        ],
    )
    print(result)
    ragas_result = result.to_pandas()
    ragas_result.to_csv(os.path.join(args.output_dir, 'ragas_score_' + args.file_name.split('.')[0] + '.csv'), index=False)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--document_path', type=str, default='../data/pdf')
    cli_parser.add_argument('--file_name', type=str, default='신여비교통비.csv')
    cli_parser.add_argument('--qa_size', type=int, default=10)
    cli_parser.add_argument('--output_dir', type=str, default='../data/pdf/output')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)