import json
import pandas as pd
from openai import OpenAI
import template
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from model_chat import examination_for_hadm
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertModel

api_key="sk-or7KMMp2Sv6GW4kA764e522a453d49E689006dA00a9a0d90"
base_url="https://api.holdai.top/v1"

class ChatBot:
    def __init__(self, api_key, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        
    def chat(self, message):
        print("Sending message to chatbot: ", message)
        messages = []
        while True:
            messages.append({"role": "user", "content": message})
            try:   
                completion = self.client.chat.completions.create(
                    model = "gpt-4o-mini",
                    messages = messages
                )
                return completion.choices[0].message.content
            except:
                print("Error, retrying...")



def major_eval_extract(major_eval):
    match = re.search(r'\b(yes|no)\b', major_eval, re.IGNORECASE)

    if match:
        if match.group(1).lower() == 'yes':
            return 1
        elif match.group(1).lower() == 'no':
            return 0
    return None
    

# chatbot = ChatBot(api_key, base_url)
# model = SentenceTransformer('../lms_embed/MedEmbed')
# diag_all = pd.read_csv('../data/heart/heart_diagnoses_all.csv')
# diag = pd.read_csv('../data/heart/heart_diagnoses.csv')
# lab = pd.read_csv('../data/heart/heart_labevents_first_lab.csv')
# micro = pd.read_csv('../data/heart/heart_microbiologyevents_first_micro.csv')
# acc_pattern = re.compile(r"Accuracy:\s*(\d)")
# com_pattern = re.compile(r"Comprehensiveness:\s*(\d)")
# safe_pattern = re.compile(r"Safety:\s*(\d)")
# with open('./result/heart_result_llama8b_0-50_RAG.json', 'r', encoding='utf-8') as f:
#     result = json.load(f)
# result_new = []
# for item in tqdm(result[0:500], total=500):
#     for key, value in item.items():
#         if value[0]['diagnosis'] == 'defeat':
#             continue
#         diag_hadm = diag_all[(diag_all['hadm_id'] == int(key))].reset_index(drop=True)
#         major_eval = chatbot.chat(template.major_eval_temp.format(diagnosis=diag_hadm['long_title'].values[0], result=value[0]['diagnosis']))
#         if diag_hadm.shape[0] >= 7:       
#             acc_eval = chatbot.chat(template.acc_eval_temp.format(diagnosis=', '.join(diag_hadm['long_title'].values[0:7].tolist()), result=value[0]['diagnosis']))
#             com_eval = chatbot.chat(template.com_eval_temp.format(diagnosis=', '.join(diag_hadm['long_title'].values[0:7].tolist()), result=value[0]['diagnosis']))
#             safe_eval = chatbot.chat(template.safe_eval_temp.format(diagnosis=', '.join(diag_hadm['long_title'].values[0:7].tolist()), result=value[0]['diagnosis']))
#         else:
#             acc_eval = chatbot.chat(template.acc_eval_temp.format(diagnosis=', '.join(diag_hadm['long_title'].values[0:diag_hadm.shape[0]].tolist()), result=value[0]['diagnosis']))
#             com_eval = chatbot.chat(template.com_eval_temp.format(diagnosis=', '.join(diag_hadm['long_title'].values[0:diag_hadm.shape[0]].tolist()), result=value[0]['diagnosis']))
#             safe_eval = chatbot.chat(template.safe_eval_temp.format(diagnosis=', '.join(diag_hadm['long_title'].values[0:diag_hadm.shape[0]].tolist()), result=value[0]['diagnosis']))
#         value[0]['major_eval'] = major_eval_extract(major_eval)
#         if acc_eval:
#             acc_score = acc_pattern.search(acc_eval)
#             if acc_score:
#                 value[0]['acc_eval'] = int(acc_score.group(1))
#             else:
#                 value[0]['acc_eval'] = None
#         if com_eval:
#             com_score = com_pattern.search(com_eval)
#             if com_score:
#                 value[0]['com_eval'] = int(com_score.group(1))
#             else:
#                 value[0]['com_eval'] = None
#         if safe_eval:
#             safe_score = safe_pattern.search(safe_eval)
#             if safe_score:
#                 value[0]['safe_eval'] = int(safe_score.group(1))
#             else:
#                 value[0]['safe_eval'] = None
#         result_list = [item.strip() for item in value[0]['diagnosis'].split(',')]
#         major_diag = [diag_hadm['long_title'].values[0]]
#         all_diag = major_diag + result_list
#         embeddings = model.encode(all_diag)
#         similarities = model.similarity(embeddings[0:1], embeddings[1:])
#         value[0]['similarity'] = float(similarities.max())
#         value[0]['exam'] = examination_for_hadm(int(key), diag, lab ,micro)
#         result_new.append(item)
# with open('./eval_result/heart_result_llama8b_0-50_RAG.json', 'w') as f:
#     json.dump(result_new, f, indent=4)

##############################################################################
# major_eval = []
# acc_eval = []
# com_eval = []
# safe_eval = []
# similiarity = []
# diag_all = pd.read_csv('../data/heart/heart_diagnoses_all.csv')
# diag = pd.read_csv('../data/heart/heart_diagnoses.csv')
# lab = pd.read_csv('../data/heart/heart_labevents_first_lab.csv')
# with open('./eval_result/heart_result_llama8b_0-50_RAG.json', 'r') as f:
#     result_new = json.load(f)
#     for item in tqdm(result_new, total=len(result_new)):
#         for key, value in item.items():
#             diag_hadm = diag_all[(diag_all['hadm_id'] == int(key))].reset_index(drop=True)
#             value[0]['true_diag'] = diag_hadm['long_title'].values.tolist()
#             major_eval.append(value[0]['major_eval'])
#             acc_eval.append(value[0]['acc_eval'])
#             com_eval.append(value[0]['com_eval'])
#             safe_eval.append(value[0]['safe_eval'])
#             similiarity.append(value[0]['similarity'])
# with open('./eval_result/heart_result_llama8b_0-50_RAG_true.json', 'w') as f:
#     json.dump(result_new, f, indent=4)
# print('major_eval', np.mean(major_eval))
# print('acc_eval', np.mean(acc_eval))
# print('com_eval', np.mean(com_eval))
# print('safe_eval', np.mean(safe_eval))
# print('similiarity', np.mean(similiarity))

##############################################################################
tokenizer = BertTokenizer.from_pretrained('../lms_embed/MedEmbed/')
model = BertModel.from_pretrained('../lms_embed/MedEmbed/', device_map='auto', trust_remote_code=True)
def calculate_matching_disease_count(model, tokenizer, list_1, list_2, threshold=0.7):

    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    def compute_similarity(embedding1, embedding2):
        return cosine_similarity([embedding1], [embedding2])[0][0]

    list_1_embeddings = [get_bert_embedding(disease) for disease in list_1]
    list_2_embeddings = [get_bert_embedding(disease) for disease in list_2]

    matching_count = 0
    list_1_matched = [False] * len(list_1)  # 标记list_1中的元素是否已匹配
    list_2_matched = [False] * len(list_2)  # 标记list_2中的元素是否已匹配

    for i, emb1 in enumerate(list_1_embeddings):
        if list_1_matched[i]:  # 如果list_1中的元素已匹配，则跳过
            continue
        for j, emb2 in enumerate(list_2_embeddings):
            if list_2_matched[j]:  # 如果list_2中的元素已匹配，则跳过
                continue
            similarity = compute_similarity(emb1, emb2)
            if similarity >= threshold:
                matching_count += 1
                print(f"匹配: {list_1[i]} <-> {list_2[j]}, 相似度: {similarity:.4f}")
                list_1_matched[i] = True  # 标记list_1中的元素已匹配
                list_2_matched[j] = True  # 标记list_2中的元素已匹配
                break  # 找到匹配后跳出内层循环，继续处理list_1中的下一个元素

    return matching_count

diag_all = pd.read_csv('../data/heart/heart_diagnoses_all_true.csv')
diag = pd.read_csv('../data/heart/heart_diagnoses.csv')
lab = pd.read_csv('../data/heart/heart_labevents_first_lab.csv')
micro = pd.read_csv('../data/heart/heart_microbiologyevents_first_micro.csv')

score_l = []
total_score_l = []
clinical_pathway_len = []
wrong_exam_len = []
exam_for_hadm_len = []
disease_count = {}
examination_count = {}
false_examination_count = {}
true_examination_count = {}
diag_count = []
with open('./result/heart_result_4o_0-2000.json', 'r') as f:
    result_new = json.load(f)
    for item in tqdm(result_new, total=len(result_new)):
        for key, value in item.items():
            diag_hadm = diag_all[(diag_all['hadm_id'] == int(key))].reset_index(drop=True)
            # value[0]['true_diag'] = diag_hadm['long_title'].values.tolist()
            # value[0]['major_diag'] = diag_hadm['long_title'].values[0:1].tolist()
            value[0]['icd_10_3'] = str(diag_hadm['icd_code'].values[0:1].tolist()[0])[0:3]
            if value[0]['diagnosis']:
                result_list = [item.strip() for item in value[0]['diagnosis'].split(',')]
            else:
                result_list = []  
            diag_count.extend(result_list)
            # score = calculate_matching_disease_count(model, tokenizer, result_list, value[0]['major_diag'])
            # total_score = calculate_matching_disease_count(model, tokenizer, result_list, list(set(value[0]['true_diag'])))
            # value[0]['matching_count'] = score
            # score_l.append(score)
            # total_score_l.append(total_score)
            
            # clinical_pathway_len.append(len(value[0]['clinical_pathway']))
            # exam_for_hadm_len.append(len(exam_for_hadm))
            # wrong_exam_len.append(len(value[0]['model_clinical_pathway'])-len(value[0]['clinical_pathway']))

            # if value[0]['icd_10_3'] not in disease_count:
            #     disease_count[value[0]['icd_10_3']] = []
            #     disease_count[value[0]['icd_10_3']].append(score)
            # else:
            #     disease_count[value[0]['icd_10_3']].append(score)
            # if value[0]['icd_10_3'] not in examination_count:
            #     examination_count[value[0]['icd_10_3']] = []
            #     examination_count[value[0]['icd_10_3']].extend(value[0]['clinical_pathway'])
            # else:
            #     examination_count[value[0]['icd_10_3']].extend(value[0]['clinical_pathway'])
            # if score == 0:
            #     if value[0]['icd_10_3'] not in false_examination_count:
            #         false_examination_count[value[0]['icd_10_3']] = []
            #         false_examination_count[value[0]['icd_10_3']].extend(value[0]['clinical_pathway'])
            #     else:
            #         false_examination_count[value[0]['icd_10_3']].extend(value[0]['clinical_pathway'])
            # if score == 1:
            #     if value[0]['icd_10_3'] not in true_examination_count:
            #         true_examination_count[value[0]['icd_10_3']] = []
            #         true_examination_count[value[0]['icd_10_3']].extend(value[0]['clinical_pathway'])
            #     else:
            #         true_examination_count[value[0]['icd_10_3']].extend(value[0]['clinical_pathway'])

# print(sum(score_l))
# print(sum(total_score_l))
# print(np.mean(exam_for_hadm_len))
# print(np.mean(clinical_pathway_len))
# print(np.mean(wrong_exam_len))
print(len(diag_count))
# for key, value in disease_count.items():
#     print(key, np.sum(value))

# def count_dict_to_df(count_dict, count_name):
#     records = []
#     for key, values in count_dict.items():
#         series_exam = pd.Series(values)
#         count_series = series_exam.value_counts()
#         for element, count in count_series.items():
#             records.append({
#                 'key': key,
#                 'element': element,
#                 count_name: count
#             })
#     return pd.DataFrame(records)

# # 转换每个计数字典
# df_exam = count_dict_to_df(examination_count, 'examination_count')
# df_false = count_dict_to_df(false_examination_count, 'false_examination_count')
# df_true = count_dict_to_df(true_examination_count, 'true_examination_count')

# # 合并DataFrame
# df_combined = pd.merge(df_exam, df_false, on=['key', 'element'], how='outer')
# df_combined = pd.merge(df_combined, df_true, on=['key', 'element'], how='outer')
# df_combined = df_combined.fillna(0).astype({
#     'examination_count': int,
#     'false_examination_count': int,
#     'true_examination_count': int
# })

# df_combined.to_excel('./result/examination_counts_combined.xlsx', index=False)
