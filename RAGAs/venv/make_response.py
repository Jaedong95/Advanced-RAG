import pandas as pd 
import os 
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import TextStreamer, GenerationConfig
from dotenv import load_dotenv

load_dotenv()
openai_api = os.getenv('OPENAI_API_KEY')

def gen(x):
    generation_config = GenerationConfig(
        temperature=0.8,
        top_p=0.8,
        top_k=100,
        max_new_tokens=1024,
        early_stopping=True,
        do_sample=True,
    )
    q = f"[INST]{x} [/INST]"
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # streamer=streamer,
    )
    result_str = tokenizer.decode(gened[0])
    start_tag = f"\n\n### Response: "
    start_index = result_str.find(start_tag)
    if start_index != -1:
        result_str = result_str[start_index + len(start_tag):].strip()
    return result_str

def main(args):
    global model 
    global tokenizer 
    global g_device
    model_name='davidkim205/komt-mistral-7b-v1'
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map='auto')
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda:0")
    g_device = torch.device("cuda") if torch.cuda.is_available() else "cpu"  
    model.to(g_device)

    gen_config = GenerationConfig(
        temperature=0.8,
        do_sample=True,
        top_p=0.95,
        max_new_tokens=512,
    )

    corpus_df = pd.read_csv(os.path.join(args.output_dir, 'qa_' + args.file_name.split('.')[0] + '.csv'), engine='pyarrow')
    print(corpus_df.head(3))    
    query_list = corpus_df['query'].values.tolist()
    print(query_list)
    response_list = [] 
    for query in query_list:
        response_list.append(gen(query).split('[/INST]')[1])
    print(response_list)
    corpus_df['response'] = response_list
    corpus_df.to_csv(os.path.join(args.output_dir, 'response_' + args.file_name.split('.')[0] + '.csv'), index=False)
    # corpus_df = corpus_df[['retrieval_gt', 'qid', 'query', 'generation']]
    # corpus_df.columns = ['retrieval_gt', 'qid', 'question', 'context', 'answer', ]

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--document_path', type=str, default='../data/pdf')
    cli_parser.add_argument('--file_name', type=str, default='경조금지급규정.csv')
    cli_parser.add_argument('--qa_size', type=int, default=10)
    cli_parser.add_argument('--output_dir', type=str, default='../data/pdf/output')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)


'''
user_query = ["회사 입사시 제출해야 하는 서류가 뭐야 ?", "내가 결혼하게 되면 회사에서 축의금을 얼마까지 지원받을 수 있어?", "외주인력 출장비 정산 방법을 알려줘"]
answer = ["회사 입사시 제출해야 하는 서류로는 이력서 1부, 자시소개서 1부, 경력증명서 (경력 사원에 한함) 등이 있습니다", \
"회사에서는 당사자가 결혼할 경우 축의금을 50만원까지 지원하며, 대표 이사의 권한으로 추가 지급할 수 있습니다.",\
"외주인력 출장비는 실비로 신청 및 정산하는 경우 실비로 지출하신 영수증을 첨부하여 지출결의서 양식으로 정산할 수 있습니다."]
metadata = ["q1", "q2", "q3"]
response_list = []; 
for query in user_query:
    model_output = gen(user_query)
    response = model_output.split('[/INST]')[1]
    print(model_output.split('[/INST]')[1])
    response_list.append(response)

evolution_type = ['simple' for _ in range(len(user_query))]
episod_done = ["True" for _ in range(len(user_query))]
# print(evolution_type) 
# print(np.dtype(response))
# print(gen(user_query))
# df = pd.DataFrame()'''