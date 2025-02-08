import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import json
# import math
import tqdm
import template
import pandas as pd
import model_chat as mc
from sentence_transformers import SentenceTransformer

diag = pd.read_csv('../data/heart/heart_diagnoses.csv')
lab = pd.read_csv('../data/heart/heart_labevents_first_lab.csv')
micro = pd.read_csv('../data/heart/heart_microbiologyevents_first_micro.csv')
diag_all = pd.read_csv('../data/heart/heart_diagnoses_all.csv')

with open('./RAG_data/PHI.json', 'r', encoding='utf-8') as f:
    PHI_data = json.load(f)

with open('./RAG_data/RAG_data_summary.json', 'r', encoding='utf-8') as f:
    summary_data = json.load(f)

embed_model = SentenceTransformer('../lms_embed/bce-embedding-base_v1/', device='cuda')
model_path = "../llms/llama3.1-8b-instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
# model = PeftModel.from_pretrained(model, "./mmed_9000_7b/12000_lora", torch_dtype=torch.bfloat16)
model.generation_config = GenerationConfig.from_pretrained(model_path)
model.to("cuda")
model.eval()

# api_key="sk-or7KMMp2Sv6GW4kA764e522a453d49E689006dA00a9a0d90", base_url="https://api.holdai.top/v1"

result_all = []
for row in tqdm.tqdm(diag[500:2000].iterrows(), total=1500):
    list_ = []
    result = mc.multi_dialogue_RAG(model, row, diag, lab, micro, diag_all, tokenizer=tokenizer, HPI_data=PHI_data, summary_data=summary_data, embed_model=embed_model)
    list_.append({"diagnosis": result[0][0],"treatment": result[0][1], "model_clinical_pathway": result[1], "clinical_pathway": result[2]})
    result_all.append({row[1]['hadm_id']: list_.copy()})
    print(result)

with open('./result/heart_result_llama8b_500-2000_RAG.json', 'w') as f:
    json.dump(result_all, f)