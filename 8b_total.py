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

model_path = "../llms/llama3.1-8b-instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
# model = PeftModel.from_pretrained(model, "./mmed_9000_7b/12000_lora", torch_dtype=torch.bfloat16)
model.generation_config = GenerationConfig.from_pretrained(model_path)
model.to("cuda")
model.eval()


result_all = []
for row in tqdm.tqdm(diag[0:1000].iterrows(), total=1000):
    list_ = []
    result = mc.total_information_diagnose(model, row, diag, lab, micro, diag_all, tokenizer=tokenizer)
    list_.append({"diagnosis": result[0],"treatment": result[1]})
    result_all.append({row[1]['hadm_id']: list_.copy()})
    print(result)

with open('./result/heart_result_llama8b_0-2000_total_v2.json', 'w') as f:
    json.dump(result_all, f)