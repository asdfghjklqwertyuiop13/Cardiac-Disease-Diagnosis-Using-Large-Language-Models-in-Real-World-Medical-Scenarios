import pandas as pd
import template
import re
from openai import OpenAI
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import json
from scipy.spatial.distance import cosine

micro_test = {
    'STOOL': 'STOOL culture',
    'URINE': 'URINE culture',
    'FLUID,OTHER': 'FLUID,OTHER',
    'BLOOD CULTURE': 'BLOOD CULTURE',
    'PLEURAL FLUID': 'PLEURAL FLUID',
    'FLUID RECEIVED IN BLOOD CULTURE BOTTLES': 'FLUID RECEIVED IN BLOOD CULTURE BOTTLES',
    'SEROLOGY/BLOOD': 'SEROLOGY/BLOOD',
    'MRSA SCREEN': 'MRSA SCREEN',
    'SPUTUM': 'SPUTUM',
    'Staph aureus swab': 'Staph aureus swab',
    'Rapid Respiratory Viral Screen & Culture': 'Rapid Respiratory Viral Screen & Culture',
    'Blood (CMV AB)': 'Blood (CMV AB)',
    'Blood (EBV)': 'Blood (EBV)',
    'Influenza A/B by DFA': 'Influenza A/B by DFA',
    'ASPIRATE': 'ASPIRATE',
    'SWAB': 'SWAB',
    'BRONCHOALVEOLAR LAVAGE': 'BRONCHOALVEOLAR LAVAGE',
    'CATHETER TIP-IV': 'CATHETER TIP-IV',
    'BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)': 'BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)',
    'Immunology (CMV)': 'Immunology (CMV)',
    'Blood (Toxo)': 'Blood (Toxo)',
    'SKIN SCRAPINGS': 'SKIN SCRAPINGS',
    'IMMUNOLOGY': 'IMMUNOLOGY',
    'PERITONEAL FLUID': 'PERITONEAL FLUID',
    'FECAL SWAB': 'FECAL SWAB',
    'FOREIGN BODY': 'FOREIGN BODY',
    'THROAT FOR STREP': 'THROAT FOR STREP',
    'TISSUE': 'TISSUE',
    'Blood (LYME)': 'Blood (LYME)',
    'JOINT FLUID': 'JOINT FLUID',
    'BILE': 'BILE',
    'DIALYSIS FLUID': 'DIALYSIS FLUID',
    'ABSCESS': 'ABSCESS',
    'BRONCHIAL WASHINGS': 'BRONCHIAL WASHINGS',
    'THROAT CULTURE': 'THROAT CULTURE',
    'CSF;SPINAL FLUID': 'CSF;SPINAL FLUID',
    'DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS': 'DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS',
    'Direct Antigen Test for Herpes Simplex Virus Types 1 & 2': 'Direct Antigen Test for Herpes Simplex Virus Types 1 & 2',
    'Mini-BAL': 'Mini-BAL',
    'BRONCHIAL BRUSH': 'BRONCHIAL BRUSH',
    'EAR': 'EAR',
    'CRE Screen': 'CRE Screen',
    'FOOT CULTURE': 'FOOT CULTURE',
    }


examination_catagory = {
    "Anemia Panel": [
        "Anemia",
        "Hematocrit", 
        "Hemoglobin", 
        "MCH", 
        "MCHC", 
        "MCV", 
        "Platelet Count", 
        "RDW", 
        "RDW-SD",
        "Red Blood Cells", 
        "White Blood Cells",
        "Basophils", 
        "Eosinophils", 
        "Lymphocytes", 
        "Monocytes", 
        "Neutrophils", 
        "Absolute Lymphocyte Count", 
        "Absolute Basophil Count", 
        "Absolute Eosinophil Count", 
        "Absolute Monocyte Count", 
        "Absolute Neutrophil Count",
        "WBC Count",
        "Lymphocytes, Percent",
        "Granulocyte Count", 
        "Eosinophil Count", 
        "Ferritin",  # 铁相关指标
        "Iron",      # 铁相关指标
        "Iron Binding Capacity, Total",  # 铁结合能力
        "Transferrin",  # 转铁蛋白
        "TIBC",
    ],
    "Cardiac Markers": [
        "Creatine Kinase ", 
        "Creatine Kinase, MB Isoenzyme", 
        "Troponin T", 
        "Troponin",
        "CK-MB Index", 
        "CK-MB",
        "BNP",
        "B-type Natriuretic Peptide",
        "NTproBNP", 
        'cardiac biomarkers',
        "Lactate Dehydrogenase"
    ],
    "Urinalysis":[
        'Urine test',
        'Urine Analysis',
    ],
    'Paracentesis':[
        'Ascites',
        'Paracentesis',
    ],
    'Bone Marrow':[
        'Bone Marrow',
    ],
    "pleural fluid":[
        "pleural fluid",
        'Thoracentesis',
    ],
    "ABG":[
        "ABG",
        "Blood Gas",
        "Calcium, ionized level",
    ],
    "Complete Blood Count (CBC)": [
        "Hematocrit", 
        "Hemoglobin", 
        "MCH", 
        "MCHC", 
        "MCV", 
        "Platelet Count", 
        "RDW", 
        "RDW-SD",
        "Red Blood Cells", 
        "White Blood Cells",
        "Basophils", 
        "Eosinophils", 
        "Lymphocytes", 
        "Monocytes", 
        "Neutrophils", 
        "Insulin resistance assessment",
        "Absolute Lymphocyte Count", 
        "Absolute Basophil Count", 
        "Absolute Eosinophil Count", 
        "Absolute Monocyte Count", 
        "Absolute Neutrophil Count",
        "WBC Count",
        "Lymphocytes, Percent",
        "Granulocyte Count", 
        "Eosinophil Count", 
        "Large Platelets",
        'Immature Granulocytes',
        'CBC',
        'Complete Blood Count',
    ],
    "BMP":[
        "Basic metabolic panel",
        "Sodium",
        "Serum calcium",
        "Potassium", 
        "Calcium, Total",
        "Urea Nitrogen",
        "Creatinine", 
        "Bicarbonate", 
        "Glucose",
        "Chloride", 
    ],
    "Comprehensive metabolic panel":[
        "CMP",
        "Sodium",
        "Serum calcium",
        "Potassium", 
        "Calcium, Total",
        "Urea Nitrogen",
        "Creatinine", 
        "Bicarbonate", 
        "Glucose",
        "Chloride", 
        "Alanine Aminotransferase (ALT)", 
        "Asparate Aminotransferase (AST)", 
        "Alkaline Phosphatase", 
        "Bilirubin, Direct", 
        "Bilirubin, Indirect", 
        "Bilirubin, Total", 
        "Albumin", 
        "Total Protein", 
        "Gamma Glutamyltransferase",
        'Haptoglobin',
        'Protein, Total',
        'Globulin',
        "Ferritin",
        'Ammonia',
        "Creatinine, Whole Blood",
        "Urea Nitrogen",
        "Creatinine", 
    ],
    "Reticulocyte Count": [
        "Reticulocyte Count, Automated", 
        "Reticulocyte Count, Absolute", 
        "Reticulocyte Count, Manual",
        'Reticulocyte Count',
        'Reticulocyte',
    ],
    "Blood Cell Morphology": [
        "Platelet Smear", 
        "RBC Morphology", 
        "Acanthocytes", 
        "Basophilic Stippling", 
        "Target Cells", 
        "Pappenheimer Bodies", 
        "Fragmented Cells", 
        "Plasma Cells", 
        "Promyelocytes", 
        "Macrocytes", 
        "Microcytes", 
        "Elliptocytes", 
        "Spherocytes", 
        "Sickle Cell Preparation", 
        "Poikilocytosis", 
        "Schistocytes", 
        "Teardrop Cells", 
        "Echinocytes", 
        "Pencil Cells", 
        "Howell-Jolly Bodies", 
        "Immature Granulocytes", 
        "Anisocytosis", 
        "Ovalocytes", 
        "Myelocytes", 
        "Nucleated Red Cells", 
        "Atypical Lymphocytes",
        "Blasts", "Metamyelocytes", 
        "Polychromasia", "Hypochromia", "Young Cells", 
        "MacroOvalocytes", "Platelet Clumps", "Bite Cells", 
        "Other Cells",
        "Bands", 
        "Hypersegmented Neutrophils", 
        "Immature Granulocytes",
        'Sickle Cells',
        'Morphology',
        'Blood smear',
    ],
    "Liver Function Test": [
        "Alanine Aminotransferase (ALT)", 
        "Asparate Aminotransferase (AST)", 
        "Alkaline Phosphatase", 
        "Bilirubin, Direct", 
        "Bilirubin, Indirect", 
        "Bilirubin, Total", 
        "Albumin", 
        "Total Protein", 
        "Gamma Glutamyltransferase", 
        "Hepatitis B Surface Antibody", 
        "Hepatitis B Surface Antigen", 
        "Hepatitis B Virus Core Antibody", 
        "Hepatitis C Virus Antibody", 
        "Hepatitis A Virus Antibody", 
        "Hepatitis B Core Antibody, IgM", 
        "Hepatitis A Virus IgM Antibody", 
        "Hepatitis C Viral Load", 
        "Hepatitis B Viral Load", 
        "Anti-Smooth Muscle Antibody", 
        "Anti-Mitochondrial Antibody",
        'Haptoglobin',
        'Protein, Total',
        'Globulin',
        'Globulin',
        "Ferritin",
        'Ammonia',
        'Liver Function',
    ],
    "Renal Function Test": [
        "Magnesium", 
        "Phosphate", 
        "Anion Gap", 
        "Albumin", 
        "Estimated GFR (MDRD equation)", 
        "Creatinine, Whole Blood", 
        "Sodium, Whole Blood", 
        "Potassium, Whole Blood", 
        "Sodium",
        "Potassium", 
        "Calcium, Total",
        "Calcium, Free", 
        "Calculated Bicarbonate, Whole Blood", 
        "Chloride", 
        "Osmolality, Measured",
        "Urea Nitrogen",
        "Creatinine", 
        "Bicarbonate", 
        "Glucose",
        "Uric Acid", 
        'Protein, Total',
        'Cryoglobulin',
        'Renal Function',
    ],
    "Thyroid Function Test": [
        "Thyroid Stimulating Hormone", 
        'Thyroid-stimulating hormone',
        "Thyroxine", 
        "Triiodothyronine", 
        "Thyroid Peroxidase Antibodies", 
        "Thyroglobulin", 
        "Anti-Thyroglobulin Antibodies",
        "Tissue Transglutaminase Ab, IgA", 
        "Parathyroid Hormone",
        'Thyroid Function',
        'parathyroid',
        'PTH',
        'TSH',
    ],
    "Coagulation and Hemostasis": [
        "INR(PT)", 
        "PT", 
        "PTT", 
        "Fibrinogen, Functional", 
        "D-Dimer", 
        "Lupus Anticoagulant", 
        "Thrombin", 
        "Factor VIII", 
        "Factor IX", 
        "Factor X", 
        "Factor VII", 
        "Factor II", 
        "Protein C, Antigen", 
        "Protein C, Functional", 
        "Protein S, Antigen", 
        "Protein S, Functional", 
        "Von Willebrand Factor Activity", 
        "Von Willebrand Factor Antigen", 
        "Antithrombin", 
        "Inhibitor Screen",
        "Factor XI", "Factor V", "Heparin", "Heparin, LMW", 
        "FMC-7", "RFXLDLM", "HIT-Ab Interpreted Result", 
        "HIT-Ab Numerical Result", "Fibrin Degradation Products",
        'Coagulation',
    ],
    "Infection and Immunology": [
        "CRP",
        "ESR",
        'Sedimentation Rate',
        'Erythrocyte Sedimentation Rate',
        "C-Reactive Protein",
        'Infection', 
        "Procalcitonin", 
        "HIV Screen", 
        "HIV 1 Viral Load", 
        "HIV 1 Ab Confirmation", 
        "HIV 2 Ab Confirmation",  
        "Epstein-Barr Virus EBNA IgG Ab", 
        "Epstein-Barr Virus VCA IgG Ab", 
        "Epstein-Barr Virus IgG Ab Value", 
        "Epstein-Barr Virus VCA IgM Ab", 
        "Epstein-Barr Virus Interpretation", 
        "Lyme C6 Ab", 
        "Toxoplasma IgG Ab", 
        "Toxoplasma IgG Ab Value", 
        "Toxoplasma Interpretation", 
        "Hepatitis C Virus Antibody", 
        "Anti-Nuclear Antibody", 
        "Anti-Nuclear Antibody, Titer", 
        "Double Stranded DNA", 
        "Rheumatoid Factor", 
        "Anti-Neutrophil Cytoplasmic ", 
        "Anti-DGP (IgA/IgG)", 
        "Cytomegalovirus IgM", 
        "Hepatitis A Virus IgM Antibody", 
        "Hepatitis B Surface Antibody", 
        "Hepatitis C Viral Load",
        "Ferritin",
        'Immature Granulocytes',
        'ANCA Titer',
        'HLA-DR',
        "C3",
        'C4',
        'Cryoglobulin',
        'Autoimmune panel',
        'ANA',
        "ANCA",
    ],
    "Pancreatic Function Test":[
        "Amylase",
        "Lipase",
    ],
    "Vitamin and Nutritional Studies": [
        "Vitamin B12", 
        "Folate", 
        "25-OH Vitamin D", 
        "Iron", 
        "Iron Binding Capacity, Total", 
        "Transferrin", 
        "Ferritin", 
        "Calculated Free Testosterone", 
        "Sex Hormone Binding Globulin", 
        "Testosterone",
        'Homocysteine',
        "Vitamin",
    ],
    "Tumor Markers": [
        "Prostate Specific Antigen", 
        "PSA",
        "Prostate-specific antigen",
        "Alpha-Fetoprotein", 
        "Carcinoembryonic Antigen (CEA)", 
        "Cancer Antigen 27.29", 
        "CA-125", 
        "Tumor marker",
        "Beta-2 Microglobulin",
        'Carcinoembyronic Antigen (CEA)',
    ],
    "Electrolyte and Acid-Base Balance": [
        "pCO2", 
        "pH", 
        "pO2", 
        "Oxygen Saturation", 
        "Base Excess", 
        "Calculated Total CO2", 
        "Required O2", 
        "Oxygen", 
        "pCO2", 
        "pO2", 
        "Oxygen Saturation", 
        "Base Excess", 
        "Calculated Total CO2",
        "lactate",
        'Electrolyte',
        "Magnesium", 
        "Phosphate", 
        "Anion Gap",
        "Sodium, Whole Blood", 
        "Potassium, Whole Blood", 
        "Sodium",
        "Potassium", 
        "Calcium, Total",
        "Calcium, Free", 
        "Calculated Bicarbonate, Whole Blood", 
        "Chloride", 
        "Osmolality, Measured",
    ],
    "Drug Monitoring": [
        "Digoxin", 
        "Vancomycin", 
        "Gentamicin", 
        "Tobramycin", 
        "Acetaminophen", 
        "Salicylate", 
        "Barbiturate Screen", 
        "Benzodiazepine Screen", 
        "Tricyclic Antidepressant Screen", 
        "Phenytoin", 
        "Phenytoin, Free", 
        "Phenytoin, Percent Free", 
        "Lithium", 
        "Methotrexate", 
        'tacroFK',  
        'Cyclosporin',  
        'Rapamycin', 
        'Amikacin',  
        "Valproic Acid",
        "Substance Abuse",
        "Opioid Withdrawal",
        "Toxicology Screen",
    ],
    "Blood Type and Antibody Screening": [ 
        "Hemoglobin A2", 
        "Hemoglobin C", 
        "Hemogloblin A", 
        "Hemogloblin S", 
    ],
    "Immunoglobulins": [
        "Immunoglobulin A", 
        "Immunoglobulin G", 
        "Immunoglobulin M", 
        "Free Kappa", 
        "Free Kappa/Free Lambda Ratio", 
        "Free Lambda", 
        "Kappa", 
        "Lambda",
        'Protein Electrophoresis',
    ],
    "Lipid Panel":[
        'Cholesterol, Total',
        'Cholesterol, LDL, Measured',
        'Cholesterol, HDL',  
        'Cholesterol Ratio (Total/HDL)',  
        'Triglycerides',
        'cholesterol',
        'Cholesterol',
        "Cholesterol, LDL, Calculated",
        "Lipid",
    ],
    "Immunophenotyping": [
        "CD20", 
        "CD20 %", 
        "CD3 %", 
        "CD4/CD8 Ratio", 
        "CD3 Absolute Count", 
        "CD4 Absolute Count",
        "CD8 Cells, Percent",
        "CD19", 
        "CD19 %", 
        "CD19 Absolute Count",
        "CD4 Cells, Percent", 
        "CD3 Cells, Percent", 
        "CD5", 
        "CD5 %", 
        "CD5 Absolute Count", 
        "CD23", 
        "CD56", 
        "CD16/56%",
        "CD16/56 Absolute Count", 
        "CD14", 
        "CD13", 
        "CD15", 
        "CD33", 
        "CD34", 
        "CD45", 
        "CD117", 
        "CD7", 
        "CD2",
        'Absolute CD4 Count',
        'CD20 Absolute Count',
        'CD64',
        'CD10',
        'Absolute CD8 Count',
        'Absolute CD3 Count',
    ],
    #病毒学
    "Virology": [
        "CMV IgG Ab", 
        "CMV IgG Ab Value", 
        "CMV IgM Ab", 
        "CMV Interpretation", 
        "Epstein-Barr Virus EBNA IgG Ab", 
        "Epstein-Barr Virus IgM Ab Value", 
        "Lyme G and M Value",
        "Cytomegalovirus Viral Load",
    ],
    "HbA1c": [
        "% Hemoglobin A1c",
        "eAG",
    ],
    "Thrombophilia Panel":[
        'Anticardiolipin Antibody IgG',
        'Anticardiolipin Antibody IgM',
        'Homocysteine', 
    ],
    "Pulmonary function test":[
        "Pulmonary function test",
    ],
    "EEG":[
        "EEG",
        "Electroencephalogram",
    ],
    "Stress Test":[
        "Stress Test",
    ]
}

exams = {
    "X-ray": [
        'CXR', 
        'XRAY', 
        'X-ray', 
        'x-ray', 
        'Xray',
        'CXRs',
        'Chest radiograph', 
        'CHEST (PORTABLE AP)', 
        'CHEST (PRE-OP PA & LAT)', 
        'CHEST (PA & LAT)',
        'CHEST XRAY PA/Lateral', 
        'Chest X-Ray', 
        'Chest Xray', 
        'Chest Film', 
        'FRONTAL CHEST RADIOGRAPH',
        'HAND XRAY', 
        'Knee XRAY', 
        'PA/LAT CXR', 
        'Chest (Portable AP)', 
        'KUB','PA and Lateral',
        '(PA & LAT)',
    ],
    "CT": [
        'CTA', 
        'CT',
        'CT A/P', 
        'CT Abdomen/Pelvis', 
        'CT Abdomen/Pelvis w/contrast', 
        'CT CHEST', 
        'CT Chest with Contrast',
        'CT Chest without Contrast', 
        'CT HEAD', 
        'CT HEAD W/O CONTRAST', 
        'CT PELVIS', 
        'CT ABD PELVIS',
        'CAROTID SERIES',
        'Head CT', 
        'HEAD CT',
        'Pulm CTA-bilat',
        "CAC",
        "Coronary Artery Calcium",
        
    ],
    "Ultrasound": [
        'US', 
        'U/S', 
        'Ultrasonography', 
        'ULTRASOUND', 
        'Ultrasound', 
        'ultrasound',
        'ECHO', 
        'Echo', 
        'echo',
        'Echocardiogram', 
        'echocardiogram',
        'ECHOCARDIOGRAPHY', 
        'LENIS', 
        'LENIs',
        'TEE', 
        'TTE',
        'ABD US', 
        'ABDOMINAL ULTRASOUND', 
        'Abdominal Ultrasound', 
        'Carotid US', 
        'Carotid Ultrasound',
        'CAROTID DOPPLER',
        'LIVER OR GALLBLADDER US', 
        'LUE Ultrasound', 
        'Left femoral US for pseudoaneurysm', 
        'Lower Extremity Ultrasound', 
        'Cardiac Echocardiogram', 
        'Carotid U/S',
        "Ejection Fraction",
    ],
    "CATH": [
        # 单词检查
        'C-cath',
        'c-cath'
        'CATH',
        'cath',
        'RHC',
        'LHC',
        # 多词检查
        'cardiac cath', 
        'CATHETERIZATION', 
        'catheterization',
        'Catheterization',
        'Coronary Angiography', 
        'RIGHT HEART AND CORONARY ARTERIOGRAPHY',
        'CARDIAC STRUCTURE/MORPH',
        'CORONARY ANGIOGRAM', 
        "Coronary Artery Angiogram",
        'coronary angiogram',
        'CARDIAC CATHETERIZATION',
        'Ventriculography',
        "Electrophysiological study",
        "EPS",
        "Endomyocardial Biopsy",
    ],
    "MRI":[
        'CARDIAC MR',
        'MRI',
        "CMR",
        'Cardiac magnetic resonance',
        'MRA',
        'Magnetic Resonance',
    ],
    "ECG":[
        'ECG', 
        "EKG", 
        'ETT', 
        'ECGStudy',
        'Continuous telemetry monitoring',
        'Holter monitor',
        'Continuous cardiac monitoring',
        'Electrocardiogram',
    ],
}

pattern_summary = r'(?<=Summary:)(.*?)(?=Next recommended examination)'

def generate_exams(exams, examination_catagory):
    exams_reversed = {}
    for key, value in exams.items():
        for item in value:
            exams_reversed[item] = key
    examination_catagory_reversed = {}
    for key, value in examination_catagory.items():
        for item in value:
            examination_catagory_reversed[item] = key
    return examination_catagory_reversed, exams_reversed

examination_catagory_reversed, exams_reversed = generate_exams(exams, examination_catagory)

def generate_response(tokenizer, model, message, messages, api_key, base_url):
    if api_key:
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
        while True:
            messages.append({"role": "user", "content": message})
            try:   
                completion = client.chat.completions.create(
                    model = model,
                    messages = messages
                )
                messages.append({"role": "assistant", "content": completion.choices[0].message.content})
                return completion.choices[0].message.content, messages
            except:
                print("Error")
    messages.append({"role": "user", "content": message })
    prompt = tokenizer.apply_chat_template(messages, return_tensors='pt').to("cuda")
    outputs = model.generate(prompt, max_new_tokens=1024, num_return_sequences=1)
    response_text = tokenizer.decode(outputs[0][len(prompt[0]):], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": response_text})
    return response_text, messages


def generate_model_input(model_input:list,flag, clinical_pathway, exam_for_hadm=None, final_diagnosis=None):
    if "physical examination" in clinical_pathway:
        if flag == 0:
            if exam_for_hadm:
                return template.first_temp_RAG.format(chief_complaint=model_input[0],phi=model_input[1],exam_for_hadm=exam_for_hadm, final_diagnosis=final_diagnosis)
            else:
                return template.first_temp.format(chief_complaint=model_input[0],phi=model_input[1])
        elif flag == 1:
            if exam_for_hadm:
                return template.existed_temp_RAG.replace("physical examination,","").format(exam=model_input[0],content=str(model_input[1]),exam_for_hadm=exam_for_hadm)
            else:
                return template.existed_temp.replace("physical examination,","").format(exam=model_input[0],content=str(model_input[1]))
        elif flag == 3:
            if exam_for_hadm:
                return template.unexisted_temp_RAG.replace("physical examination,","").format(exam=model_input[0],exam_for_hadm=exam_for_hadm)
            else:
                return template.unexisted_temp.replace("physical examination,","").format(exam=model_input[0])
        elif flag == 2:
            return template.summary_temp.replace("physical examination,","").format(exam=model_input[0],content=str(model_input[1]))
        elif flag == 4:
            if exam_for_hadm:
                return template.been_chosen_temp_RAG.replace("physical examination,","").format(exam=model_input[0],exam_for_hadm=exam_for_hadm)
            else:
                return template.been_chosen_temp.replace("physical examination,","").format(exam=model_input[0])
        elif flag == 5:
            return template.final_temp
    else:
        if flag == 0:
            if exam_for_hadm:
                return template.first_temp_RAG.format(chief_complaint=model_input[0],phi=model_input[1],exam_for_hadm=exam_for_hadm,final_diagnosis=final_diagnosis)
            else:
                return template.first_temp.format(chief_complaint=model_input[0],phi=model_input[1])
        elif flag == 1:
            if exam_for_hadm:
                return template.existed_temp_RAG.replace("physical examination,","").format(exam=model_input[0],content=str(model_input[1]),exam_for_hadm=exam_for_hadm)
            else:
                return template.existed_temp.replace("physical examination,","").format(exam=model_input[0],content=str(model_input[1]))
        elif flag == 3:
            if exam_for_hadm:
                return template.unexisted_temp_RAG.replace("physical examination,","").format(exam=model_input[0],exam_for_hadm=exam_for_hadm)
            else:
                return template.unexisted_temp.replace("physical examination,","").format(exam=model_input[0])
        elif flag == 2:
            return template.summary_temp.replace("physical examination,","").format(exam=model_input[0],content=str(model_input[1]))
        elif flag == 4:
            if exam_for_hadm:
                return template.been_chosen_temp_RAG.replace("physical examination,","").format(exam=model_input[0],exam_for_hadm=exam_for_hadm)
            else:
                return template.been_chosen_temp.replace("physical examination,","").format(exam=model_input[0])
        elif flag == 5:
            return template.final_temp

def output_process(output, model_clinical_pathway, clinical_pathway, examination_catagory_reversed, exams_reversed):
    pattern = r"Next recommended examination\s*(.*?)($)"
    diag_pattern = r"Final diagnosis:\s*(.*?)(?=Treatment:|$)"
    treat_pattern = r"Treatment:\s*(.*)"
    
    diag = re.search(diag_pattern, output.replace("*", ""), re.DOTALL|re.IGNORECASE)
    treat= re.search(treat_pattern, output.replace("*", ""), re.DOTALL|re.IGNORECASE)

    if diag:
        if treat:
            return (diag.group(1), treat.group(1))
        else:
            return (diag.group(1), None)
    else:
        recommended_examination = re.search(pattern, output.replace("*", ""), re.DOTALL|re.IGNORECASE)
        if recommended_examination:
            # print(recommended_examination.group(1))
            words = recommended_examination.group(1)
            model_clinical_pathway.append(words)

            if "Physical examination".lower() in recommended_examination.group(1).lower():
                if "physical examination" in model_clinical_pathway:
                    model_clinical_pathway[-1] = "physical examination"
                    return ["physical examination", 1, model_clinical_pathway]
                model_clinical_pathway[-1] = "physical examination"
                clinical_pathway.append("physical examination")
                return ["physical examination", clinical_pathway, model_clinical_pathway]
            
            for key,value in exams_reversed.items():
                if len(value) >= 5:
                    if value.lower() in recommended_examination.group(1).lower():
                        if value in model_clinical_pathway:
                            model_clinical_pathway[-1] = value
                            return [value, 1, model_clinical_pathway]
                        model_clinical_pathway[-1] = value
                        clinical_pathway.append(value)
                        return [value, clinical_pathway, model_clinical_pathway] 
                else:
                    if value in recommended_examination.group(1):
                        if value in model_clinical_pathway:
                            model_clinical_pathway[-1] = value
                            return [value, 1, model_clinical_pathway]
                        model_clinical_pathway[-1] = value
                        clinical_pathway.append(value)
                        return [value, clinical_pathway, model_clinical_pathway]
                    
            for key,value in examination_catagory_reversed.items():
                if len(value) >= 5:
                    if value.lower() in recommended_examination.group(1).lower():
                        if value in model_clinical_pathway:
                            model_clinical_pathway[-1] = value
                            return [value, 1, model_clinical_pathway]
                        model_clinical_pathway[-1] = value
                        clinical_pathway.append(value)
                        return [value, clinical_pathway, model_clinical_pathway] 
                else:
                    if value in recommended_examination.group(1):
                        if value in model_clinical_pathway:
                            model_clinical_pathway[-1] = value
                            return [value, 1, model_clinical_pathway]
                        model_clinical_pathway[-1] = value
                        clinical_pathway.append(value)
                        return [value, clinical_pathway, model_clinical_pathway]
                      
            for key,value in exams_reversed.items():
                if len(key) >= 5:
                    if key.lower() in recommended_examination.group(1).lower():
                        if value in model_clinical_pathway:
                            model_clinical_pathway[-1] = value
                            return [value, 1, model_clinical_pathway]
                        model_clinical_pathway[-1] = value
                        clinical_pathway.append(value)
                        return [value, clinical_pathway, model_clinical_pathway]
                else:
                    if key in recommended_examination.group(1):
                        if value in model_clinical_pathway:
                            model_clinical_pathway[-1] = value
                            return [value, 1, model_clinical_pathway]
                        model_clinical_pathway[-1] = value
                        clinical_pathway.append(value)
                        return [value, clinical_pathway, model_clinical_pathway]
                    
            for key,value in examination_catagory_reversed.items():
                if len(key) >= 5:
                    if key.lower() in recommended_examination.group(1).lower():
                        if value in model_clinical_pathway:
                            model_clinical_pathway[-1] = value
                            return [value, 1, model_clinical_pathway]
                        model_clinical_pathway[-1] = value
                        clinical_pathway.append(value)
                        return [value, clinical_pathway, model_clinical_pathway]
                else:
                    if key in recommended_examination.group(1):
                        if value in model_clinical_pathway:
                            model_clinical_pathway[-1] = value
                            return [value, 1, model_clinical_pathway]
                        model_clinical_pathway[-1] = value
                        clinical_pathway.append(value)
                        return [value, clinical_pathway, model_clinical_pathway] 
                    
            for key,value in micro_test.items():
                if len(value) >= 5:
                    if value.lower() in recommended_examination.group(1).lower():
                        if value in model_clinical_pathway:
                            model_clinical_pathway[-1] = value
                            return [value, 1, model_clinical_pathway]
                        model_clinical_pathway[-1] = value
                        clinical_pathway.append(value)
                        return [key, clinical_pathway, model_clinical_pathway] 
                else:
                    if value in recommended_examination.group(1):
                        if value in model_clinical_pathway:
                            model_clinical_pathway[-1] = value
                            return [value, 1, model_clinical_pathway]
                        model_clinical_pathway[-1] = value
                        clinical_pathway.append(value)
                        return [key, clinical_pathway, model_clinical_pathway]
            return [words, None , "no existed examination"]
    

def choose_examination(hadm_id, value, row:tuple, lab:pd.DataFrame, micro:pd.DataFrame, model_clinical_pathway, clinical_pathway):
    if value == "physical examination":
        return [value, row[1]['physical_exam'], clinical_pathway]
    if value in exams.keys():
        if value == "ECG":
            exam = str(row[1][value]) + str(row[1]['reports'])
            if len(exam) > 2:
                return [value, exam, clinical_pathway]
            else:
                if clinical_pathway:
                    clinical_pathway.pop()
                return [value, 0, clinical_pathway]
        else:
            exam = row[1][value]
            if len(exam) > 2:
                return [value, exam, clinical_pathway]
            else:
                if clinical_pathway:
                    clinical_pathway.pop()
                return [value, 0, clinical_pathway]
    elif value in examination_catagory.keys():
        exam = lab[lab['hadm_id'] == hadm_id]
        lab_test = []
        if value == "ABG":
            exam = exam[exam['examination_group'] == "Blood Gas"]
            if not exam.empty:
                for row in exam.iterrows():
                    lab_result = str(row[1]['label']) + ' ' + str(row[1]['valuenum']) + ' ' + str(row[1]['valueuom'])
                    if pd.notna(row[1]['flag']):
                        lab_result = lab_result + ' ' + str(row[1]['flag'])
                    if pd.notna(row[1]['comments']):
                        lab_result = lab_result + ' ' + str(row[1]['comments'])
                    lab_test.append(lab_result)
                return [value, lab_test, clinical_pathway]
            else:
                if clinical_pathway:
                    clinical_pathway.pop()
                return [value, 0, clinical_pathway]
        if value == "Urinalysis":
            exam = exam[exam['examination_group'] == "Urine Test"]
            if not exam.empty:
                for row in exam.iterrows():
                    lab_result = str(row[1]['label']) + ' ' + str(row[1]['valuenum']) + ' ' + str(row[1]['valueuom'])
                    if pd.notna(row[1]['flag']):
                        lab_result = lab_result + ' ' + str(row[1]['flag'])
                    if pd.notna(row[1]['comments']):
                        lab_result = lab_result + ' ' + str(row[1]['comments'])
                    lab_test.append(lab_result)
                return [value, lab_test, clinical_pathway]
            else:
                if clinical_pathway:
                    clinical_pathway.pop()
                return [value, 0, clinical_pathway]
        if value == "pleural fluid":
            exam = exam[exam['fluid'] == "Pleural"]
            if not exam.empty:
                for row in exam.iterrows():
                    lab_result = str(row[1]['label']) + ' ' + str(row[1]['valuenum']) + ' ' + str(row[1]['valueuom'])
                    if pd.notna(row[1]['flag']):
                        lab_result = lab_result + ' ' + str(row[1]['flag'])
                    if pd.notna(row[1]['comments']):
                        lab_result = lab_result + ' ' + str(row[1]['comments'])
                    lab_test.append(lab_result)
                return [value, lab_test, clinical_pathway]
            else:
                if clinical_pathway:
                    clinical_pathway.pop()
                return [value, 0, clinical_pathway]
        if value == "Paracentesis":
            exam = exam[exam['fluid'] == "Ascites"]
            if not exam.empty:
                for row in exam.iterrows():
                    lab_result = str(row[1]['label']) + ' ' + str(row[1]['valuenum']) + ' ' + str(row[1]['valueuom'])
                    if pd.notna(row[1]['flag']):
                        lab_result = lab_result + ' ' + str(row[1]['flag'])
                    if pd.notna(row[1]['comments']):
                        lab_result = lab_result + ' ' + str(row[1]['comments'])
                    lab_test.append(lab_result)
                return [value, lab_test, clinical_pathway]
            else:
                if clinical_pathway:
                    clinical_pathway.pop()
                return [value, 0, clinical_pathway]
        if value == "Bone Marrow":
            exam = exam[exam['fluid'] == "Bone Marrow"]
            if not exam.empty:
                for row in exam.iterrows():
                    lab_result = str(row[1]['label']) + ' ' + str(row[1]['valuenum']) + ' ' + str(row[1]['valueuom'])
                    if pd.notna(row[1]['flag']):
                        lab_result = lab_result + ' ' + str(row[1]['flag'])
                    if pd.notna(row[1]['comments']):
                        lab_result = lab_result + ' ' + str(row[1]['comments'])
                    lab_test.append(lab_result)
                return [value, lab_test, clinical_pathway]
            else:
                if clinical_pathway:
                    clinical_pathway.pop()
                return [value, 0, clinical_pathway]
        for item in examination_catagory[value]:
            index = exam[exam['label'] == item]
            if not index.empty:
                lab_result = str(item) + ' ' + str(index['valuenum'].values[0]) + ' ' + str(index['valueuom'].values[0])
                if pd.notna(index['flag'].iloc[0]):
                    lab_result = lab_result + ' ' + str(index['flag'].values[0])
                if pd.notna(index['comments'].iloc[0]):
                    lab_result = lab_result + ' ' + str(index['comments'].values[0])
                lab_test.append(lab_result)
        if lab_test:
            return [value, lab_test, clinical_pathway]
        else:
            if clinical_pathway:
                clinical_pathway.pop()
            return [value, 0, clinical_pathway]
    elif value in micro_test.keys():
        inter = {"S": 'sensitive', 'R': 'resistant', 'I':'intermediate', 'P':"pending"}
        exam = micro[micro['hadm_id'] == hadm_id]
        exam = exam[exam['spec_type_desc'] == value]
        if not exam.empty:
            for row in exam.iterrows():
                result = ""
                if pd.notna(row[1]['org_name']):
                    result += " " + str(row[1]['org_name'])
                if pd.notna(row[1]['ab_name']):
                    result += " " + str(row[1]['ab_name'])
                if pd.notna(row[1]['interpretation']):
                    result += " " + inter[str(row[1]['interpretation'])]
                if pd.notna(row[1]['comments']):
                    result += " " + str(row[1]['comments'])
                if result == "":
                    result = "NO GROWTH."
            return [micro_test[value], result, clinical_pathway]
        else:
            if clinical_pathway:
                clinical_pathway.pop()
            return [micro_test[value], 0, clinical_pathway]
    else:
        if clinical_pathway:
                clinical_pathway.pop()
        return [value, 0, clinical_pathway]
    

def multi_dialogue(model, row:tuple,diag:pd.DataFrame, lab:pd.DataFrame, micro:pd.DataFrame, tokenizer=None, api_key=None, base_url=None, HPI_data=None, summary_data=None):
    flag = 0
    turn = 0
    repeated = 0
    messages = []
    model_clinical_pathway = []
    clinical_pathway = []
    while True:
        # input("程序已暂停，按回车继续...")
        print("model_clinical_pathway:",model_clinical_pathway)
        print("flag:", flag)
        print("repeated", repeated)
        if repeated >= 3:
            flag = 5
        if turn == 2:
            flag = 2
        if flag == 0:#first
            model_input = [row[1]['chief_complaint'],row[1]['HPI']]
            message = generate_model_input(model_input,flag, clinical_pathway)
            response_text, messages = generate_response(tokenizer, model, message, messages, api_key, base_url)
            output = output_process(response_text, model_clinical_pathway, clinical_pathway,examination_catagory_reversed, exams_reversed)
            print('*'*100)
            print('model_input:',message)
            print('*'*100)
            print("response_text:",response_text)
            print('*'*100)
            print("output_process:",output)
            if not output:
                return [("defeat", None), model_clinical_pathway, clinical_pathway]
            if type(output) == list:
                if type(output[1]) == list:
                    model_clinical_pathway = output[2]
                    clinical_pathway = output[1]
                    exam = choose_examination(row[1]['hadm_id'], output[0],row, lab, micro, model_clinical_pathway,clinical_pathway)
                    clinical_pathway = exam[2]
                    if exam[1] == 0:
                        turn = 0
                        flag = 3
                    else:
                        turn += 1
                        flag = 1
                elif output[1] == 1:
                    model_clinical_pathway = output[2]
                    exam = output
                    turn += 1
                    repeated += 1
                    flag = 4
                elif output[2] == "no existed examination":
                    exam = output
                    turn = 0
                    flag = 3
            else:
                return [output, model_clinical_pathway, clinical_pathway]
        elif flag == 1:#existed
            message = generate_model_input(exam,flag, clinical_pathway)
            response_text, messages = generate_response(tokenizer, model, message, messages, api_key, base_url)
            output = output_process(response_text, model_clinical_pathway,clinical_pathway,examination_catagory_reversed, exams_reversed)
            print('*'*100)
            print('model_input:',message)
            print('*'*100)
            print("response_text:",response_text)
            print('*'*100)
            print("output_process:",output)
            if not output:
                return [("defeat", None), model_clinical_pathway, clinical_pathway]
            if type(output) == list:
                if type(output[1]) == list:
                    model_clinical_pathway = output[2]
                    clinical_pathway = output[1]
                    exam = choose_examination(row[1]['hadm_id'], output[0],row, lab, micro, model_clinical_pathway,clinical_pathway)
                    clinical_pathway = exam[2]
                    if exam[1] == 0:
                        turn = 0
                        flag = 3
                    else:
                        turn += 1
                        flag = 1
                elif output[1] == 1:
                    model_clinical_pathway = output[2]
                    exam = output
                    repeated += 1
                    turn += 1
                    flag = 4
                elif output[2] == "no existed examination":
                    exam = output
                    turn = 0
                    flag = 3
            else:
                return [output, model_clinical_pathway, clinical_pathway]
        elif flag == 3:#unexisted
            message = generate_model_input(exam,flag, clinical_pathway)
            response_text, messages = generate_response(tokenizer, model, message, messages, api_key, base_url)
            output = output_process(response_text, model_clinical_pathway, clinical_pathway, examination_catagory_reversed, exams_reversed)
            print('*'*100)
            print('model_input:',message)
            print('*'*100)
            print("response_text:",response_text)
            print('*'*100)
            print("output_process:",output)
            if not output:
                return [("defeat", None), model_clinical_pathway, clinical_pathway]
            if type(output) == list:
                if type(output[1]) == list:
                    model_clinical_pathway = output[2]
                    clinical_pathway = output[1]
                    exam = choose_examination(row[1]['hadm_id'], output[0],row, lab, micro, model_clinical_pathway,clinical_pathway)
                    clinical_pathway = exam[2]
                    if exam[1] == 0:
                        turn = 0
                        flag = 3
                    else:
                        flag = 1
                elif output[1] == 1:
                    model_clinical_pathway = output[2]
                    exam = output
                    repeated += 1
                    turn += 1
                    flag = 4
                elif output[2] == "no existed examination":
                    exam = output
                    turn = 0
                    flag = 3
            else:
                return [output, model_clinical_pathway, clinical_pathway]
        elif flag == 2:#summary
            message = generate_model_input(exam,flag, clinical_pathway)
            response_text, messages = generate_response(tokenizer, model, message, messages, api_key, base_url)
            output = output_process(response_text, model_clinical_pathway, clinical_pathway, examination_catagory_reversed, exams_reversed)
            print('*'*100)
            print('model_input:',message)
            print('*'*100)
            print("response_text:",response_text)
            print('*'*100)
            print("output_process:",output)
            if not output:
                return [("defeat", None), model_clinical_pathway, clinical_pathway]
            if type(output) == list:
                if type(output[1]) == list:
                    model_clinical_pathway = output[2]
                    clinical_pathway = output[1]
                    exam = choose_examination(row[1]['hadm_id'], output[0],row, lab, micro, model_clinical_pathway,clinical_pathway)
                    clinical_pathway = exam[2]
                    if exam[1] == 0:
                        turn = 0
                        flag = 3
                    else:
                        turn += 1
                        flag = 1
                elif output[1] == 1:
                    model_clinical_pathway = output[2]
                    exam = output
                    repeated += 1
                    turn += 1
                    flag = 4
                elif output[2] == "no existed examination":
                    exam = output
                    turn = 0
                    flag = 3
            else:
                return [output, model_clinical_pathway, clinical_pathway]
            
        elif flag == 4:#been_chosen
            message = generate_model_input(exam,flag, clinical_pathway)
            response_text, messages = generate_response(tokenizer, model, message, messages, api_key, base_url)
            output = output_process(response_text, model_clinical_pathway, clinical_pathway, examination_catagory_reversed, exams_reversed)
            print('*'*100)
            print('model_input:',message)
            print('*'*100)
            print("response_text:",response_text)
            print('*'*100)
            print("output_process:",output)
            if not output:
                return [("defeat", None), model_clinical_pathway, clinical_pathway]
            if type(output) == list:
                if type(output[1]) == list:
                    model_clinical_pathway = output[2]
                    clinical_pathway = output[1]
                    exam = choose_examination(row[1]['hadm_id'], output[0],row, lab, micro, model_clinical_pathway,clinical_pathway)
                    clinical_pathway = exam[2]
                    if exam[1] == 0:
                        turn = 0
                        flag = 3
                    else:
                        turn += 1
                        flag = 1
                elif output[1] == 1:
                    model_clinical_pathway = output[2]
                    exam = output
                    repeated += 1
                    turn += 1
                    flag = 4
                elif output[2] == "no existed examination":
                    exam = output
                    turn = 0
                    flag = 3
            else:
                return [output, model_clinical_pathway, clinical_pathway]
        elif flag == 5:#repeated
            message = generate_model_input(exam,flag, clinical_pathway)
            response_text, messages = generate_response(tokenizer, model, message, messages, api_key, base_url)
            output = output_process(response_text, model_clinical_pathway, clinical_pathway, examination_catagory_reversed, exams_reversed)
            print('*'*100)
            print('model_input:',message)
            print('*'*100)
            print("response_text:",response_text)
            print('*'*100)
            print("output_process:",output)
            if not output:
                return [("defeat", None), model_clinical_pathway, clinical_pathway]
            if type(output) == list:
                if type(output[1]) == list:
                    model_clinical_pathway = output[2]
                    clinical_pathway = output[1]
                    exam = choose_examination(row[1]['hadm_id'], output[0],row, lab, micro, model_clinical_pathway,clinical_pathway)
                    clinical_pathway = exam[2]
                    if exam[1] == 0:
                        turn = 0
                        flag = 3
                    else:
                        turn += 1
                        flag = 1
                elif output[1] == 1:
                    model_clinical_pathway = output[2]
                    exam = output
                    repeated += 1
                    turn += 1
                    flag = 4
                elif output[2] == "no existed examination":
                    exam = output
                    turn = 0
                    flag = 3
            else:
                return [output, model_clinical_pathway, clinical_pathway]

def retrieve_best_match(model, query_text, data):
    texts = [list(item.values())[0] for item in data]
    keys = [list(item.keys())[0] for item in data]
    with ThreadPoolExecutor() as executor:
        text_vectors = list(executor.map(model.encode, texts))
    query_vector = model.encode([query_text])[0]
    similarities = [1 - cosine(query_vector, text_vector) for text_vector in text_vectors]
    # print(max(similarities))
    best_match_idx = np.argmax(similarities)
    best_match_key = keys[best_match_idx]
    return best_match_key

def examination_for_hadm(hadm_id, diag:pd.DataFrame, lab:pd.DataFrame, micro:pd.DataFrame):
    exam = []
    diag_hadm = diag[diag['hadm_id'] == hadm_id].reset_index(drop=True)
    lab_hadm = lab[(lab['hadm_id'] == hadm_id) & (lab['fluid'] != 'Other Body Fluid')].reset_index(drop=True)
    micro_hadm = micro[micro['hadm_id'] == hadm_id].reset_index(drop=True)
    if not diag_hadm.empty:
        exam.append("physical examination")
        for row in diag_hadm.iterrows():
            if len(row[1]['X-ray']) >= 3:
                exam.append('X-ray')
            if len(row[1]['CT']) >= 3:
                exam.append('CT')
            if len(row[1]['Ultrasound']) >= 3:
                exam.append('Ultrasound')
            if len(row[1]['MRI']) >= 3:
                exam.append('MRI')
            exam.append('ECG')
    if not lab_hadm.empty:
        for row in lab_hadm.iterrows():
            if row[1]['examination_group'] == 'Blood Gas':
                exam.append("ABG")
            elif row[1]['examination_group'] == 'Urine Test':
                exam.append("Urinalysis")
            elif row[1]['fluid'] == 'Pleural':
                exam.append("pleural fluid")
            elif row[1]['fluid'] == 'Ascites':
                exam.append("Paracentesis")
            elif row[1]['fluid'] == 'Bone Marrow':
                exam.append("Bone Marrow")
            elif row[1]['examination_group'] in examination_catagory.keys():
                exam.append(row[1]['examination_group'])
    if not micro_hadm.empty:
        for row in micro_hadm.iterrows():
            if row[1]['spec_type_desc'] in micro_test:
                exam.append(micro_test[row[1]['spec_type_desc']])
    unique_list = []
    seen = set()
    for item in exam:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list

def multi_dialogue_RAG(model, row:tuple,diag:pd.DataFrame, lab:pd.DataFrame, micro:pd.DataFrame, diag_all:pd.DataFrame, tokenizer=None, api_key=None, base_url=None, HPI_data=None, summary_data=None, embed_model=None):
    flag = 0
    turn = 0
    repeated = 0
    messages = []
    model_clinical_pathway = []
    clinical_pathway = []
    while True:
        # input("程序已暂停，按回车继续...")
        print("model_clinical_pathway:",model_clinical_pathway)
        print("flag:", flag)
        print("repeated", repeated)
        if repeated >= 3:
            flag = 5
        if turn == 2:
            flag = 2
        if flag == 0:#first
            model_input = [row[1]['chief_complaint'],row[1]['HPI']]
            most_similar_HPI_hadm = retrieve_best_match(embed_model, row[1]['HPI'], HPI_data)
            exam_for_hadm = examination_for_hadm(int(most_similar_HPI_hadm), diag, lab, micro)
            final_diagnosis = diag_all[diag_all['hadm_id']==int(most_similar_HPI_hadm)]['long_title'].values[0:7].tolist()
            message = generate_model_input(model_input,flag, clinical_pathway, exam_for_hadm=exam_for_hadm, final_diagnosis=final_diagnosis)
            response_text, messages = generate_response(tokenizer, model, message, messages, api_key, base_url)
            output = output_process(response_text, model_clinical_pathway, clinical_pathway,examination_catagory_reversed, exams_reversed)
            print('*'*100)
            print('model_input:',message)
            print('*'*100)
            print("response_text:",response_text)
            print('*'*100)
            print("output_process:",output)
            exam_for_hadm = []
            if not output:
                return [("defeat", None), model_clinical_pathway, clinical_pathway]
            if type(output) == list:
                if type(output[1]) == list:
                    model_clinical_pathway = output[2]
                    clinical_pathway = output[1]
                    exam = choose_examination(row[1]['hadm_id'], output[0],row, lab, micro, model_clinical_pathway,clinical_pathway)
                    clinical_pathway = exam[2]
                    if exam[1] == 0:
                        turn = 0
                        flag = 3
                    else:
                        turn += 1
                        flag = 1
                elif output[1] == 1:
                    model_clinical_pathway = output[2]
                    exam = output
                    turn += 1
                    repeated += 1
                    flag = 4
                elif output[2] == "no existed examination":
                    exam = output
                    turn = 0
                    flag = 3
            else:
                return [output, model_clinical_pathway, clinical_pathway]
        elif flag == 1:#existed
            message = generate_model_input(exam ,flag ,clinical_pathway , exam_for_hadm=list(filter(lambda x: x not in clinical_pathway, exam_for_hadm)))
            response_text, messages = generate_response(tokenizer, model, message, messages, api_key, base_url)
            output = output_process(response_text, model_clinical_pathway,clinical_pathway,examination_catagory_reversed, exams_reversed)
            print('*'*100)
            print('model_input:',message)
            print('*'*100)
            print("response_text:",response_text)
            print('*'*100)
            print("output_process:",output)
            exam_for_hadm = []
            if not output:
                return [("defeat", None), model_clinical_pathway, clinical_pathway]
            if type(output) == list:
                if type(output[1]) == list:
                    model_clinical_pathway = output[2]
                    clinical_pathway = output[1]
                    exam = choose_examination(row[1]['hadm_id'], output[0],row, lab, micro, model_clinical_pathway,clinical_pathway)
                    clinical_pathway = exam[2]
                    if exam[1] == 0:
                        turn = 0
                        flag = 3
                    else:
                        turn += 1
                        flag = 1
                elif output[1] == 1:
                    model_clinical_pathway = output[2]
                    exam = output
                    repeated += 1
                    turn += 1
                    flag = 4
                elif output[2] == "no existed examination":
                    exam = output
                    turn = 0
                    flag = 3
            else:
                return [output, model_clinical_pathway, clinical_pathway]
        elif flag == 3:#unexisted
            message = generate_model_input(exam,flag, clinical_pathway, exam_for_hadm=list(filter(lambda x: x not in clinical_pathway, exam_for_hadm))) 
            response_text, messages = generate_response(tokenizer, model, message, messages, api_key, base_url)
            output = output_process(response_text, model_clinical_pathway, clinical_pathway, examination_catagory_reversed, exams_reversed)
            print('*'*100)
            print('model_input:',message)
            print('*'*100)
            print("response_text:",response_text)
            print('*'*100)
            print("output_process:",output)
            exam_for_hadm = []
            if not output:
                return [("defeat", None), model_clinical_pathway, clinical_pathway]
            if type(output) == list:
                if type(output[1]) == list:
                    model_clinical_pathway = output[2]
                    clinical_pathway = output[1]
                    exam = choose_examination(row[1]['hadm_id'], output[0],row, lab, micro, model_clinical_pathway,clinical_pathway)
                    clinical_pathway = exam[2]
                    if exam[1] == 0:
                        turn = 0
                        flag = 3
                    else:
                        flag = 1
                elif output[1] == 1:
                    model_clinical_pathway = output[2]
                    exam = output
                    repeated += 1
                    turn += 1
                    flag = 4
                elif output[2] == "no existed examination":
                    exam = output
                    turn = 0
                    flag = 3
            else:
                return [output, model_clinical_pathway, clinical_pathway]
        elif flag == 2:#summary
            message = generate_model_input(exam,flag, clinical_pathway)
            response_text, messages = generate_response(tokenizer, model, message, messages, api_key, base_url)
            output = output_process(response_text, model_clinical_pathway, clinical_pathway, examination_catagory_reversed, exams_reversed)
            print('*'*100)
            print('model_input:',message)
            print('*'*100)
            print("response_text:",response_text)
            print('*'*100)
            print("output_process:",output)
            if not output:
                return [("defeat", None), model_clinical_pathway, clinical_pathway]
            if type(output) == list:
                if type(response_text) == str:
                    summary = re.search(pattern_summary, response_text.replace("*", ""), flags=re.DOTALL|re.IGNORECASE)
                    if summary:
                        exam_for_hadm = examination_for_hadm(int(retrieve_best_match(embed_model, summary.group(1), summary_data)) ,diag, lab ,micro)
                if type(output[1]) == list:
                    model_clinical_pathway = output[2]
                    clinical_pathway = output[1]
                    exam = choose_examination(row[1]['hadm_id'], output[0],row, lab, micro, model_clinical_pathway,clinical_pathway)
                    clinical_pathway = exam[2]
                    if exam[1] == 0:
                        turn = 0
                        flag = 3
                    else:
                        turn += 1
                        flag = 1
                elif output[1] == 1:
                    model_clinical_pathway = output[2]
                    exam = output
                    repeated += 1
                    turn += 1
                    flag = 4
                elif output[2] == "no existed examination":
                    exam = output
                    turn = 0
                    flag = 3
            else:
                return [output, model_clinical_pathway, clinical_pathway]
            
        elif flag == 4:#been_chosen
            message = generate_model_input(exam,flag, clinical_pathway, exam_for_hadm=list(filter(lambda x: x not in clinical_pathway, exam_for_hadm)))
            response_text, messages = generate_response(tokenizer, model, message, messages, api_key, base_url)
            output = output_process(response_text, model_clinical_pathway, clinical_pathway, examination_catagory_reversed, exams_reversed)
            print('*'*100)
            print('model_input:',message)
            print('*'*100)
            print("response_text:",response_text)
            print('*'*100)
            print("output_process:",output)
            exam_for_hadm = []
            if not output:
                return [("defeat", None), model_clinical_pathway, clinical_pathway]
            if type(output) == list:
                if type(output[1]) == list:
                    model_clinical_pathway = output[2]
                    clinical_pathway = output[1]
                    exam = choose_examination(row[1]['hadm_id'], output[0],row, lab, micro, model_clinical_pathway,clinical_pathway)
                    clinical_pathway = exam[2]
                    if exam[1] == 0:
                        turn = 0
                        flag = 3
                    else:
                        turn += 1
                        flag = 1
                elif output[1] == 1:
                    model_clinical_pathway = output[2]
                    exam = output
                    repeated += 1
                    turn += 1
                    flag = 4
                elif output[2] == "no existed examination":
                    exam = output
                    turn = 0
                    flag = 3
            else:
                return [output, model_clinical_pathway, clinical_pathway]
        elif flag == 5:#repeated
            message = generate_model_input(exam,flag, clinical_pathway)
            response_text, messages = generate_response(tokenizer, model, message, messages, api_key, base_url)
            output = output_process(response_text, model_clinical_pathway, clinical_pathway, examination_catagory_reversed, exams_reversed)
            print('*'*100)
            print('model_input:',message)
            print('*'*100)
            print("response_text:",response_text)
            print('*'*100)
            print("output_process:",output)
            if not output:
                return [("defeat", None), model_clinical_pathway, clinical_pathway]
            if type(output) == list:
                if type(output[1]) == list:
                    model_clinical_pathway = output[2]
                    clinical_pathway = output[1]
                    exam = choose_examination(row[1]['hadm_id'], output[0],row, lab, micro, model_clinical_pathway,clinical_pathway)
                    clinical_pathway = exam[2]
                    if exam[1] == 0:
                        turn = 0
                        flag = 3
                    else:
                        turn += 1
                        flag = 1
                elif output[1] == 1:
                    model_clinical_pathway = output[2]
                    exam = output
                    repeated += 1
                    turn += 1
                    flag = 4
                elif output[2] == "no existed examination":
                    exam = output
                    turn = 0
                    flag = 3
            else:
                return [output, model_clinical_pathway, clinical_pathway]



def total_information_diagnose(model, row:tuple,diag:pd.DataFrame, lab:pd.DataFrame, micro:pd.DataFrame, diag_all:pd.DataFrame, tokenizer=None, api_key=None, base_url=None):
    HPI_information = row[1]['HPI']
    chief_complaint = row[1]['chief_complaint']
    # HPI_input = template.HPI_summary.format(chief_complaint=chief_complaint, HPI=HPI_information)
    # print(HPI_input)
    # HPI_output, _ = generate_response(tokenizer, model, HPI_input, [], api_key, base_url)
    # print(HPI_output)
    # print("*"*100)

    physical_exam_information = row[1]['physical_exam']
    # physical_exam_input = template.physicial_summary.format(physical_exam=physical_exam_information)
    # print(physical_exam_input)
    # physical_exam_output, _ = generate_response(tokenizer, model, physical_exam_input, [], api_key, base_url)
    # print(physical_exam_output)
    # print("*"*100)

    lab_exam = lab[lab["hadm_id"]==row[1]['hadm_id']]
    lab_test = []
    if not lab_exam.empty:
        for lab_row in lab_exam.iterrows():
            lab_result = str(lab_row[1]['label']) + ' ' + str(lab_row[1]['valuenum']) + ' ' + str(lab_row[1]['valueuom'])
            if pd.notna(lab_row[1]['flag']):
                lab_result = lab_result + ' ' + str(lab_row[1]['flag'])
            if pd.notna(lab_row[1]['comments']):
                lab_result = lab_result + ' ' + str(lab_row[1]['comments'])
            if pd.notna(lab_row[1]['fluid']):
                lab_result = lab_result + ' ' + str(lab_row[1]['fluid'])
            lab_test.append(lab_result)
    if lab_test:
        lab_input = template.lab_summary.format(lab_exam=lab_test)
        print(lab_input)
        lab_output, _ = generate_response(tokenizer, model, lab_input, [], api_key, base_url)
    else:
        lab_output = "no existed lab test"
    print(lab_output)
    print("*"*100)

    image_test = []
    if len(row[1]['X-ray']) > 2:
        image_test.append("X-ray:" + str(row[1]['X-ray']))
    if len(row[1]['CT']) > 2:
        image_test.append("CT:" + str(row[1]['CT']))
    if len(row[1]['Ultrasound']) > 2:
        image_test.append("Ultrasound:" + str(row[1]['Ultrasound']))
    if len(row[1]['CATH']) > 2:
        image_test.append("CATH:" + str(row[1]['CATH']))
    if len(row[1]['MRI']) > 2:
        image_test.append("MRI:" + str(row[1]['MRI']))
    if len(row[1]['ECG']) > 2:
        image_test.append("ECG:" + str(row[1]['ECG']) + str(row[1]['reports']))
    else:
        image_test.append("ECG:" + str(row[1]['reports']))
    if image_test:
        image_input = template.image_summary.format(image_exam=str(image_test))
        print(image_input)
        image_output, _ = generate_response(tokenizer, model, image_input, [], api_key, base_url)
    else:
        image_output = "no existed image test"
    print(image_output)
    print("*"*100)

    inter = {"S": 'sensitive', 'R': 'resistant', 'I':'intermediate', 'P':"pending"}
    micro_exam = micro[micro['hadm_id'] == row[1]['hadm_id']]
    micro_test = []
    if not micro_exam.empty:
        for micro_row in micro_exam.iterrows():
            result = ""
            if pd.notna(micro_row[1]['test_name']):
                result += " " + str(micro_row[1]['test_name'])
            if pd.notna(micro_row[1]['org_name']):
                result += " " + str(micro_row[1]['org_name'])
            if pd.notna(micro_row[1]['ab_name']):
                result += " " + str(micro_row[1]['ab_name'])
            if pd.notna(micro_row[1]['interpretation']):
                result += " " + inter[str(micro_row[1]['interpretation'])]
            if pd.notna(micro_row[1]['comments']):
                result += " " + str(micro_row[1]['comments'])
            if result == "":
                result = "NO GROWTH."
            micro_test.append(result)
    if micro_test:
        micro_input = template.micro_summary.format(micro_exam=str(micro_test))
        print(micro_input)
        micro_output, _ = generate_response(tokenizer, model, micro_input, [], api_key, base_url)
    else:
        micro_output = "no existed micro test"
    print(micro_output)
    print("*"*100)

    summary_input = template.summary_v2.format(chief_complaint=chief_complaint, HPI=HPI_information, physical_summary=physical_exam_information, lab_summary=lab_output, image_summary=image_output, micro_summary=micro_output)
    print(summary_input)
    summary_output, _ = generate_response(tokenizer, model, summary_input, [], api_key, base_url)
    print(summary_output)
    print("*"*100)

    diag_pattern = r"Final diagnosis:\s*(.*?)(?=Treatment:|$)"
    treat_pattern = r"Treatment:\s*(.*)"
    if type(summary_output) == str:
        result_diag = re.search(diag_pattern, summary_output.replace("*", ""), re.DOTALL)
        result_treat= re.search(treat_pattern, summary_output.replace("*", ""), re.DOTALL)
        if result_diag:
            if result_treat:
                return (result_diag.group(1), result_treat.group(1))
            else:
                return (result_diag.group(1), None)
        else:
            return (None, None)
    else:
        return (None, None)

    
                


    
