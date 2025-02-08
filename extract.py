import pandas as pd
from tqdm import tqdm
import re
import ast
import json

# # 定义符合条件的ICD编码范围
ranges = [
    # I20-I25
    'I20', 'I21', 'I22', 'I23', 'I24', 'I25',
    # I30-I51
    'I30', 'I31', 'I32', 'I33', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39', 
    'I40', 'I41', 'I42', 'I43', 'I44', 'I45', 'I46', 'I47', 'I48', 'I49',
    'I50',
    # 410-414
    '410', '411', '412', '413', '414',
    # # 420-429
    '420', '421', '422', '423', '424', '425', '426', '427', '428'
]


# #######################################################
# # import re

# # with open('./heart_discharge.json', 'r', encoding='utf-8') as f:
# #     data = json.load(f)

# # # 定义正则表达式匹配检查项目的模式
# # pattern = re.compile(
# #     r"""
# #     ^                              # 匹配行首
# #     (?P<exam_name>                 # 命名捕获组
# #         (?:[A-Z][A-Za-z0-9\s\/\-\(\)_]{0,100})  # 匹配检查名称，包括下划线
# #     )
# #     [\s]*                          # 匹配可能的空格
# #     (?::|\n|$)                     # 匹配冒号、换行或行尾
# #     """,
# #     re.VERBOSE | re.MULTILINE
# # )
# # list_ = []
# # # 使用正则表达式查找检查项目
# # for i in data:
# #     if i['exam'] != None:
# #         matches = pattern.findall(i['exam'])
# #         exam_list = list(set(matches))
# #         list_.extend(exam_list)

# # list_ = list(set(list_))
# # # 打印检查项目列表
# # print("检查项目列表：", list_)
# # with open('./heart_discharge_exam.json', 'w', encoding='utf-8') as f:
# #     json.dump(list_, f, ensure_ascii=False, indent=4)
# #######################################################


single_word_exams = [
    'C-cath',
    'CATH',
    'CXR',
    'CXRs',
    'CTA',
    'CT',
    'ECG',
    'EEG',
    'EKG',
    'KUB',
    'PFT',
    'TEE',
    'TTE',
    'DLCO',
    'ECHO',
    'EEG',
    'EGD',
    'Echo',
    'Echocardiogram',
    'ECHOCARDIOGRAPHY',
    'LENIS',
    'LENIs',
    'MRA',
    'US',
    'ETT',
    'RHC',
    'U/S',
    'Ultrasonography',
    'XRAY',
    'X-ray',
    'x-ray',
    'Xray',
    'ULTRASOUND',
    'Ultrasound',
    'ECGStudy',
]
multi_word_exams = [
    'ABD US',
    'ABDOMINAL ULTRASOUND',
    'Abdominal Ultrasound',
    'cardiac cath',
    'CATHETERIZATION',
    'CORONARY ANGIOGRAM',
    'CORONARY ANGIOGRAPHY',
    'CARDIAC CATHETERIZATION',
    'CARDIAC STRUCTURE/MORPH',
    'CAROTID SERIES',
    'CAROTID DOPPLER',
    'Carotid US',
    'CAROTID US',
    'Carotid Ultrasound',
    'Chest radiograph',
    'CHEST (PORTABLE AP)',
    'CHEST (PRE-OP PA & LAT)',
    'CHEST (PA & LAT)',
    '(PA & LAT)',
    'CHEST XRAY PA/Lateral',
    'CT A/P',
    'CT Abdomen/Pelvis ',
    'CT Abdomen/Pelvis w/contrast',
    'CT CHEST',
    'CT Chest with Contrast',
    'CT Chest without Contrast',
    'CT HEAD ',
    'CT HEAD W/O CONTRAST',
    'CT PELVIS',
    'CT ABD PELVIS',
    'CT Pelvis with Contrast',
    'Cardiac Echocardiogram',
    'Carotid U/S',
    'Chest Film',
    'Chest X-Ray',
    'Chest Xray',
    'CARDIAC MR',
    'Coronary Angiography',
    'FRONTAL CHEST RADIOGRAPH',
    'HAND XRAY',
    'Head CT',
    'HEAD CT',
    'Knee XRAY',
    'LIVER OR GALLBLADDER US',
    'LUE Ultrasound',
    'Left UE ultrasound',
    'Left femoral US for pseudoaneurysm',
    'Lower Extremity Ultrasound',
    'MR Head',
    'MRI HEAD',
    'MR HEAD',
    'PA and Lateral',
    'PA/LAT CXR',
    'Pulm CTA-bilat',
    'RIGHT HEART AND CORONARY ARTERIOGRAPHY',
    'CHEST (PORTABLE AP)',
    'STRESS TEST',
]
alerts=['PRE-CPB', 'POST-CPB', 'PRE BYPASS', 'POST BYPASS','POST CPB', 'PRE CPB' ,'POST-BYPASS', 'PRE-BYPASS', 'PREBYPASS', 'POSTBYPASS', 'EEG', 'EGD', 'Stress test', 'PFT', 'DLCO','MRA', 'MR Head', 'MRI HEAD', 'MR HEAD', 'CT Pelvis', 'Knee XRAY', 'CT Abd', 'CT Abdomen', 'CT HEAD', 'CT HEAD W/O CONTRAST', 'CT PELVIS', 'CT ABD PELVIS','Head CT', 'HEAD CT', 'ABD US', 'ABDOMINAL ULTRASOUND', 'Abdominal Ultrasound', 'LIVER OR GALLBLADDER US', 'Pulmonary function test']


list_ = []
def split_text(text):
    pattern_phi = r'(?<=History of Present Illness)(.*?)(?=Social History|Past Medical History|Social History|Family History|Physical Exam)'
    pattern_physical_exam = r'(.*?)(?=physical exam)'  # 匹配 physical exam 前的部分
    pattern_brief_hospital_course = r'(?=physical exam)(.*?)(?=brief hospital course|procedures)'  # 匹配 physical exam 和 brief hospital course 之间的部分
    pattern_discharge = r'discharge medications(.*?)(?=Discharge Disposition)'  # 匹配 brief hospital course 和 discharge medicine 之间的部分
    pattern_chief = r'(?<=Complaint:)(.*?)(?=Major Surgical or Invasive Procedure)'  # 匹配 chief complaint 和 history of present illness 之间的部分
    pattern_invasion = r'(?<=Major Surgical or Invasive Procedure:)(.*?)(?=History of Present Illness)'
    
    # pattern_brief_exam = r'(?=Pertinent Result)(.*?)(?=brief hospital course|procedures)'

    part0 = re.search(pattern_phi, text, re.DOTALL)
    # part1 = re.search(pattern_physical_exam, text, re.DOTALL|re.IGNORECASE)
    part2 = re.search(pattern_brief_hospital_course, text, re.DOTALL|re.IGNORECASE)
    # part3 = re.search(pattern_discharge, text, re.DOTALL|re.IGNORECASE)
    part4 = re.search(pattern_chief, text, re.DOTALL|re.IGNORECASE)
    part5 = re.search(pattern_invasion, text, re.DOTALL|re.IGNORECASE)
    # part6 = re.search(pattern_brief_exam, text, re.DOTALL|re.IGNORECASE)

    result_part0 = part0.group(1) if part0 else None
    # result_part1 = part1.group(1) if part1 else None
    result_part2 = part2.group(1) if part2 else None
    # result_part3 = part3.group(1) if part3 else None
    result_part4 = part4.group(1) if part4 else None
    result_part5 = part5.group(1) if part5 else None
    # result_part6 = part6.group(1) if part6 else None

    return result_part0 , result_part2, result_part4, result_part5

# import re
# import json

def extract_physical_reports(report):
    results = {}
    physical_pattern = re.compile(rf"(?<![A-Za-z0-9] )\b{'physical exam'}\b(?! [A-Za-z0-9]):?\s*(.*?)(?=(\n\n|\Z|\n\.\n|Pertinent|DISCHARGE))", flags=re.DOTALL|re.IGNORECASE)
    end_pattern = re.compile(r"(.*?)(?=(\n\n|Pertinent|DISCHARGE|\n.\n|rtinent))", flags=re.DOTALL)
    match = physical_pattern.search(report)
    if match:
        content = match.group(1)
        end_pos = match.start(1) + len(content)

        check_region = report[max(0, end_pos-20):end_pos]
        if ':' in check_region or '+' in check_region or '0' in check_region or '1' in check_region or '2' in check_region or '3' in check_region or '4' in check_region or '5' in check_region or '6' in check_region or '7' in check_region or '8' in check_region or '9' in check_region or "ght" in check_region.lower():
            next_match = end_pattern.search(report, end_pos+2)
            if next_match:
                content += next_match.group(1)
                end_pos = next_match.start(1) + len(next_match.group(1))
        results['physical exam'] = content.strip()    
        return content.strip(), report[end_pos:].strip()
    else:
        print("Physical exam not found.")
        return None, report

def remove_time_stamped_entries(report):
    time_stamped_pattern = r"___ \d{2}:\d{2}(AM|PM).*?(?=(\n|\Z))"
    cleaned_report = re.sub(time_stamped_pattern, "", report, flags=re.DOTALL)
    lab_line_pattern = r"(?m)^[A-Za-z0-9/()\-%\*\.\s]+-[-+0-9\.]+.*?$"
    cleaned_report = re.sub(lab_line_pattern, "", cleaned_report)
    cleaned_report = re.sub(r"\n\s*\n", "\n\n", cleaned_report)
    return cleaned_report

def split_exams(report, single_word_exams=single_word_exams, multi_word_exams=multi_word_exams, alerts=alerts):
   
    single_word_exams = sorted(single_word_exams, key=len, reverse=True)
    multi_word_exams = sorted(multi_word_exams, key=len, reverse=True)
    matches = []

    for exam in alerts:
        if exam.lower() in report.lower():
            print(f"Alert: {exam} found in report.")
            return {}
        
    for exam in multi_word_exams:
        pattern = rf"(?<!: |[a-su-z0-9] ){re.escape(exam)}|{re.escape(exam)}(?!\.|,| [a-z0-9])"
        for match in re.finditer(pattern, report, flags=re.DOTALL|re.IGNORECASE):
            matches.append((match.start(), match.end(), exam))

    for exam in single_word_exams:
        pattern = rf"(?<!: |[a-z0-9] )\b{re.escape(exam)}\b|\b{re.escape(exam)}\b(?!\.|,| [a-z0-9]|  N)"
        for match in re.finditer(pattern, report, flags=re.DOTALL):
            matches.append((match.start(), match.end(), exam))

    matches.sort(key=lambda x: x[0])
    sections = {}
    sections["X-ray"] = []
    sections["CT"] = []
    sections["Ultrasound"] = []
    sections["CATH"] = []
    sections["ECG"] = []
    sections["MRI"] = []
    for i, (start, end, exam_name) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(report)
        exam_content = report[end:next_start].strip()
        exam_content = remove_after_keywords(exam_content, ["pertinent", "DISCHARGE"])
        if exam_content:
            if exam_name in sections:
                sections[exam_name].append(exam_content)
            else:
                sections[exam_name] = [exam_content]
    return sections

exams = {
    "X-ray": [
        'CXR', 'XRAY', 'X-ray', 'x-ray', 'Xray','CXRs',
        'Chest radiograph', 'CHEST (PORTABLE AP)', 'CHEST (PRE-OP PA & LAT)', 'CHEST (PA & LAT)',
        'CHEST XRAY PA/Lateral', 'Chest X-Ray', 'Chest Xray', 'Chest Film', 'FRONTAL CHEST RADIOGRAPH',
        'HAND XRAY', 'Knee XRAY', 'PA/LAT CXR', 'Chest (Portable AP)', 'KUB','PA and Lateral','(PA & LAT)',
    ],
    "CT": [
        'CTA', 'CT',
        'CT A/P', 'CT Abdomen/Pelvis', 'CT Abdomen/Pelvis w/contrast', 'CT CHEST', 'CT Chest with Contrast',
        'CT Chest without Contrast', 'CT HEAD', 'CT HEAD W/O CONTRAST', 'CT PELVIS', 'CT ABD PELVIS','CAROTID SERIES','Head CT', 'HEAD CT','Pulm CTA-bilat',
        
    ],
    "Ultrasound": [
        'US', 'U/S', 'Ultrasonography', 'ULTRASOUND', 'Ultrasound', 'ECHO', 'Echo', 'Echocardiogram', 'ECHOCARDIOGRAPHY', 'LENIS', 'LENIs',
        'TEE', 'TTE', 'Left UE ultrasound',
        'ABD US', 'ABDOMINAL ULTRASOUND', 'Abdominal Ultrasound', 'Carotid US', 'Carotid Ultrasound','CAROTID DOPPLER',
        'LIVER OR GALLBLADDER US', 'LUE Ultrasound', 'Left femoral US for pseudoaneurysm', 'Lower Extremity Ultrasound', 'Cardiac Echocardiogram', 'Carotid U/S',
    ],
    "CATH": [
        'C-cath', 'CATH', 'RHC',
        'cardiac cath', 'CATHETERIZATION', 'Coronary Angiography', 'RIGHT HEART AND CORONARY ARTERIOGRAPHY','CARDIAC STRUCTURE/MORPH','CORONARY ANGIOGRAM', 'CARDIAC CATHETERIZATION'
    ],
    "MRI":[
        'CARDIAC MR',
    ],
    "ECG":[
        'ECG', "EKG", 'ETT', 'ECGStudy'
    ],
}



alias_to_category = {}
for category, aliases in exams.items():
    for alias in aliases:
        alias_to_category[alias.lower()] = category
    

def standardize_keys(results, alias_to_category=alias_to_category):
    standardized_results = {}
    
    for key, value in results.items():

        standardized_key = key.lower()
        
        if standardized_key in alias_to_category:
            category = alias_to_category[standardized_key]
            
            if category in standardized_results:
                standardized_results[category].extend(value)
            else:
                standardized_results[category] = value
        else:
            standardized_results[key] = value
    
    return standardized_results

def remove_after_keywords(text, keywords):
    pattern = rf"({'|'.join(map(re.escape, keywords))}).*"
    # 使用 re.sub 替换匹配的内容为空
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL|re.IGNORECASE)
    return cleaned_text
keywords = ["MICRO", "DISCHARGE"]

####################################################### examination group
examination_groups = {
    "Cardiac Markers": [
        "Creatine Kinase (CK)", 
        "Creatine Kinase, MB Isoenzyme", 
        "Troponin T", 
        "CK-MB Index", 
        "NTproBNP", 
        "Lactate Dehydrogenase (LD)"
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
    ],
    "BMP":[
        "Sodium","Potassium", "Calcium, Total","Urea Nitrogen","Creatinine", "Bicarbonate", 
        "Glucose","Chloride", 
    ],
    "Reticulocyte Count": [
        "Reticulocyte Count, Automated", 
        "Reticulocyte Count, Absolute", 
        "Reticulocyte Count, Manual",
        'Reticulocyte Count',
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
    ],
    "Liver Function Tests": [
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
    ],
    "Renal Function Tests": [
        "Magnesium", 
        "Phosphate", 
        "Anion Gap", 
        "Estimated GFR (MDRD equation)", 
        "Creatinine, Whole Blood", 
        "Sodium, Whole Blood", 
        "Potassium, Whole Blood", 
        "Calcium, Free", 
        "Uric Acid", 
        "Calculated Bicarbonate, Whole Blood", 
        "Osmolality, Measured"
    ],
    "Thyroid Function Tests": [
        "Thyroid Stimulating Hormone", 
        "Thyroxine (T4), Free", 
        "Triiodothyronine (T3)", 
        "Thyroxine (T4)", 
        "Thyroid Peroxidase Antibodies", 
        "Thyroglobulin", 
        "Anti-Thyroglobulin Antibodies",
        "Tissue Transglutaminase Ab, IgA", 
        "Parathyroid Hormone"
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
        "HIT-Ab Numerical Result", "Fibrin Degradation Products"
    ],
    "Infection and Immunology": [
        'Sedimentation Rate',
        "C-Reactive Protein", 
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
        "Anti-Neutrophil Cytoplasmic Antibody", 
        "Anti-DGP (IgA/IgG)", 
        "Cytomegalovirus IgM", 
        "Hepatitis A Virus IgM Antibody", 
        "Hepatitis B Surface Antibody", 
        "Hepatitis C Viral Load"
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
        "Testosterone"
    ],
    "Tumor Markers": [
        "Prostate Specific Antigen", 
        "Alpha-Fetoprotein", 
        "Carcinoembryonic Antigen (CEA)", 
        "Cancer Antigen 27.29", 
        "CA-125", 
        "Beta-2 Microglobulin"
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
        "lactate"
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
        "Valproic Acid"
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
        "Cholesterol, LDL, Calculated",
    ],
    "Immunophenotyping": [
        "CD20", "CD20 %", "CD3 %", "CD4/CD8 Ratio", "CD3 Absolute Count", 
        "CD4 Absolute Count",
        "CD8 Cells, Percent", "CD19", "CD19 %", "CD19 Absolute Count",
        "CD4 Cells, Percent", "CD3 Cells, Percent", "CD5", 
        "CD5 %", "CD5 Absolute Count", "CD23", "CD56", "CD16/56%",
        "CD16/56 Absolute Count", "CD14", "CD13", "CD15", 
        "CD33", "CD34", "CD45", "CD117", "CD7", "CD2"
    ],
    "Virology": [
        "CMV IgG Ab", "CMV IgG Ab Value", "CMV IgM Ab", 
        "CMV Interpretation", "Epstein-Barr Virus EBNA IgG Ab", 
        "Epstein-Barr Virus IgM Ab Value", "Lyme G and M Value","Cytomegalovirus Viral Load",
    ],
    "HbA1c": [
        "% Hemoglobin A1c",
        "eAG",
    ]
}


# 筛选条件
# df = pd.read_csv('./diagnoses_icd.csv')
# df2 = pd.read_csv('./icd9toicd10cmgem.csv')
# filterd = df[(df['seq_num'] == 1) & (df['icd_code'].str[:3].isin(ranges))]
# icd_diag = pd.read_csv('./d_icd_diagnoses.csv')
# icd9_to_icd10 = dict(zip(df2['icd9cm'], df2['icd10cm']))
# def convert_icd_code(icd_code):
#     if icd_code in icd9_to_icd10:
#         return icd9_to_icd10[icd_code]  # 替换为ICD-10编码
#     return icd_code  # 如果是ICD-10编码，或者没有映射，则保持不变
# filterd['icd_code'] = filterd['icd_code'].apply(convert_icd_code)
# filterd['icd_code_3'] = filterd['icd_code'].str[:3]
# # df.drop(columns=['long_title', 'icd_code', 'icd_code3'], inplace=True)
# merged_df = pd.merge(filterd, icd_diag, left_on='icd_code_3', right_on='icd_code', how='left')
# merged_df.drop(columns=['icd_code_x', 'icd_code_y','icd_version_x', 'icd_version_y'], inplace=True)
# merged_df.dropna(subset=['long_title'], inplace=True)
# merged_df = merged_df[merged_df["icd_code_3"].str[:3].isin(ranges)]
# merged_df.to_csv('./heart_diagnoses_10.csv', index=False)

###################################################### procedures
# diag = pd.read_csv('./heart_diagnoses.csv')
# procedure = pd.read_csv('./procedures_icd.csv')
# procedure = procedure[procedure['hadm_id'].isin(diag['hadm_id'])]
# icd_pro = pd.read_csv('./d_icd_procedures.csv')
# merged_df = pd.merge(procedure, icd_pro, left_on='icd_code', right_on='icd_code', how='left')
# merged_df.drop(columns=['icd_version_x', 'icd_version_y'], inplace=True)
# merged_df.to_csv('./heart_procedures.csv', index=False)


#######################################################
# disc = pd.read_csv('./heart_diagnoses.csv')
# data = pd.read_csv('./heart_procedures.csv')
# data = data[data['hadm_id'].isin(disc['hadm_id'])]
# data.to_csv('./heart_procedures.csv', index=False)
#######################################################
# disc = pd.read_csv('./discharge.csv')
# diag = pd.read_csv('./heart_diagnoses_10.csv')
# df = disc[disc['hadm_id'].isin(diag['hadm_id'])]
# # # df = pd.read_csv('./heart_discharge_all.csv')
# # # df = df[0:100]
# for index, row in tqdm(df.iterrows(), total=df.shape[0]):
#     phi, report, chief, invasion = split_text(row['text'])
#     if phi is not None and report is not None:
#         phy, exam = extract_physical_reports(report)
#         results = standardize_keys((split_exams(remove_time_stamped_entries(exam))))
#         df.at[index, 'HPI'] = phi
#         df.at[index, 'physical_exam'] = phy
#         df.at[index, 'chief_complaint'] = chief
#         df.at[index, 'invasions'] = invasion
#         for key, value in results.items():
#             df.at[index, key] = str(value)
# df = df[(df['physical_exam'].notna()) & (df['HPI'].notna()) & ((df['CT'].notna()) | (df['Ultrasound'].notna()) | (df['CATH'].notna()) | (df['X-ray'].notna()) | (df['ECG'].notna()))]
# cols = ['X-ray', 'CT', 'Ultrasound', 'CATH', 'ECG', 'MRI']
# df = df[df[cols].map(lambda x: len(ast.literal_eval(x)) > 0).astype(int).sum(axis=1) > 2]

# output_chunks = []
# for chunk in (pd.read_csv('./machine_measurements.csv', chunksize=100000)):
#     output = chunk[chunk['subject_id'].isin(diag['subject_id'])]
#     output_chunks.append(output)
# ecg_df = pd.concat(output_chunks, ignore_index=True)
# ecg_df['ecg_time'] = pd.to_datetime(ecg_df['ecg_time'])
# df['storetime'] = pd.to_datetime(df['storetime'])
# merged_df = pd.merge(ecg_df, df, on='subject_id', how='inner')

# one_month = pd.Timedelta(days=30)  # 近似一个月的天数
# filtered_df = merged_df[(merged_df['ecg_time'] <= merged_df['storetime']) & 
#                         (merged_df['ecg_time'] >= merged_df['storetime'] - one_month)]

# report_columns = [f'report_{i}' for i in range(18)]  # ['report_0', 'report_1', ..., 'report_17']

# # 对每个subject_id的所有report字段去重并合并
# def merge_reports(row):
#     # 获取所有非空的报告值
#     reports = row[report_columns].dropna().unique()
#     return reports

# # 对filtered_df按subject_id进行处理
# filtered_df['reports'] = filtered_df.apply(merge_reports, axis=1)

# # 按subject_id进行聚合，并去重报告，合并为一个唯一的报告列表
# def aggregate_reports(reports_list):
#     # 合并所有report列表中的报告并去重
#     combined_reports = pd.Series(reports_list).explode().dropna().unique()
#     return ' | '.join(combined_reports)  # 使用 " | " 作为报告之间的分隔符

# # 聚合每个subject_id下的报告，并生成ecg_machine字段
# ecg_machine_summary = filtered_df.groupby('subject_id')['reports'].agg(aggregate_reports).reset_index()

# # 将ecg_machine_summary合并到discharge表中
# discharge_df = pd.merge(df, ecg_machine_summary, on='subject_id', how='left')
# discharge_df.dropna(subset=['reports'], inplace=True)
# discharge_df = discharge_df[discharge_df['physical_exam'].str.len() > (len('ADMISSION PHYSICAL EXAM:\n========================\n')+5)]
# # 将结果保存到新的CSV文件
# discharge_df.to_csv('heart_discharge_all.csv', index=False)
# discharge_df.drop(['text'], axis=1, inplace=True)
# discharge_df.to_csv('heart_discharge.csv', index=False)

#######################################################

# diag = pd.read_csv('./heart_diagnoses.csv')
# for row in tqdm(diag.iterrows(), total=diag.shape[0]):
#     if len(row[1]['CT']) > 2:
#         for item in ast.literal_eval(row[1]['CT']):
#             print(remove_after_keywords(item, keywords))
#             print("*"*100)

#######################################################

# diag = pd.read_csv('./heart_diagnoses.csv')
# output_chunks = []
# for chunk in pd.read_csv('./labevents.csv', chunksize=100000):
#     output = chunk[chunk['hadm_id'].isin(diag['hadm_id'])]
#     # merged_lab = pd.merge(chunk, d_lab, left_on='itemid', right_on='itemid', how='left')
#     output_chunks.append(output)
# lab = pd.concat(output_chunks)
# lab.to_csv('./heart_labevents.csv', index=False)

#######################################################

# lab = pd.read_csv('./heart_labevents.csv')
# d_lab = pd.read_csv('./d_labitems.csv')
# lab = pd.merge(lab, d_lab, left_on='itemid', right_on='itemid', how='left')
# examination_function = {}
# for key, value in examination_groups.items():
#     for item in value:
#         examination_function[item] = key

# for row in tqdm(lab.iterrows(), total=lab.shape[0]):
#     if row[1]['fluid'] == "Blood" and row[1]['label'] in examination_function.keys():
#         lab.at[row[0], 'examination_group'] = examination_function[row[1]['label']]
#     if row[1]['category'] == 'Blood Gas':
#         lab.at[row[0], 'examination_group'] = 'Blood Gas'
#     if row[1]['fluid'] == 'Urine':
#         lab.at[row[0], 'examination_group'] = 'Urine Test'
# for row in tqdm(lab.iterrows(), total=lab.shape[0]):
#     if pd.isna(row[1]['examination_group']):
#         lab.at[row[0], 'examination_group'] = row[1]['label']
# lab.drop(['subject_id','labevent_id','itemid','order_provider_id','priority'], axis=1, inplace=True)
# lab.to_csv('./heart_labevents_examination_group.csv', index=False)
# lab['charttime'] = pd.to_datetime(lab['charttime'])
# first_tests = lab.loc[lab.groupby(['hadm_id', 'label', 'fluid'])['charttime'].idxmin()]
# first_tests = first_tests.reset_index(drop=True)
# first_tests.to_csv('./heart_labevents_first_lab.csv', index=False)

###################################################### diagnoses_all

# df2 = pd.read_csv('./icd9toicd10cmgem.csv')
# icd9_to_icd10 = dict(zip(df2['icd9cm'], df2['icd10cm']))
# def convert_icd_code(icd_code):
#     if icd_code in icd9_to_icd10:
#         return icd9_to_icd10[icd_code]  # 替换为ICD-10编码
#     return icd_code  # 如果是ICD-10编码，或者没有映射，则保持不变
# diag = pd.read_csv('./heart_diagnoses.csv')
# diag_all = pd.read_csv('./diagnoses_icd.csv')
# diag_all = diag_all[diag_all['hadm_id'].isin(diag['hadm_id'])]
# icd_diag = pd.read_csv('./d_icd_diagnoses.csv')
# diag_all['icd_code'] = diag_all['icd_code'].apply(convert_icd_code).str[:3]
# merged_df = pd.merge(diag_all, icd_diag, left_on='icd_code', right_on='icd_code', how='left')
# merged_df.drop(columns=['icd_version_x', 'icd_version_y'], inplace=True)
# merged_df.to_csv('./heart_diagnoses_all.csv', index=False)

# diag_all = pd.read_csv('./heart_diagnoses_all.csv')
# diag_cleaned = diag_all.dropna(subset=['long_title'])
# diag_cleaned = diag_cleaned.drop_duplicates(subset=['hadm_id', 'long_title'], keep='first')
# diag_cleaned.to_csv('./heart_diagnoses_all_cleaned.csv', index=False)

###################################################### micro

# diag = pd.read_csv('./heart_diagnoses.csv')
# output_chunks = []
# for chunk in pd.read_csv('./microbiologyevents.csv', chunksize=100000):
#     output = chunk[chunk['hadm_id'].isin(diag['hadm_id'])]
#     output_chunks.append(output)
# result = pd.concat(output_chunks, ignore_index=True)
# result.to_csv('./heart_microbiologyevents.csv', index=False)

# df = pd.read_csv('./heart_microbiologyevents.csv')
# df['chartdate'] = pd.to_datetime(df['chartdate'])
# first_tests = df.loc[df.groupby(['hadm_id', 'test_name'])['charttime'].idxmin()]
# first_tests = first_tests.reset_index(drop=True)
# first_tests.to_csv('./heart_microbiologyevents_first_micro.csv', index=False)

######################################################radio-details

# df = pd.read_csv('./radiology_detail.csv')
# df_filtered = df[df['field_name'] == 'exam_name']
# df_filtered.to_csv('./radiology_detail_name.csv', index=False)

#######################################################radio_merge
# df = pd.read_csv('./radiology_detail_name.csv')
# radio = pd.read_csv('./heart_radiology.csv')
# merged = pd.merge(radio, df, left_on='note_id', right_on='note_id', how='left')
# # merged.drop(columns=['field_name_x', 'field_name_y', 'field_ordinal'], inplace=True)
# merged.to_csv('./heart_radiology.csv', index=False)
# radio = pd.read_csv('./heart_radiology.csv')
# radio = radio[(radio['field_value'].str[0] != '-') & (radio['field_value'].str[0] != '_')]
# radio.to_csv('./heart_radiology.csv', index=False)

#######################################################first-radio
# radio = pd.read_csv('./heart_radiology.csv')
# radio['charttime'] = pd.to_datetime(radio['charttime'])
# first_tests = radio.loc[radio.groupby(['hadm_id', 'field_value'])['charttime'].idxmin()]
# first_tests = first_tests.reset_index(drop=True)
# first_tests.to_csv('./heart_radiology_first.csv', index=False)

# examination_catagory = {
#     "Cardiac Markers": [
#         "Creatine Kinase (CK)", 
#         "Creatine Kinase, MB Isoenzyme", 
#         "Troponin T", 
#         "CK-MB Index", 
#         "NTproBNP", 
#         "Lactate Dehydrogenase (LD)"
#     ],
#     "Complete Blood Count (CBC)": [
#         "Hematocrit", 
#         "Hemoglobin", 
#         "MCH", 
#         "MCHC", 
#         "MCV", 
#         "Platelet Count", 
#         "RDW", 
#         "RDW-SD",
#         "Red Blood Cells", 
#         "White Blood Cells",
#         "Basophils", 
#         "Eosinophils", 
#         "Lymphocytes", 
#         "Monocytes", 
#         "Neutrophils", 
#         "Absolute Lymphocyte Count", 
#         "Absolute Basophil Count", 
#         "Absolute Eosinophil Count", 
#         "Absolute Monocyte Count", 
#         "Absolute Neutrophil Count",
#         "WBC Count",
#         "Lymphocytes, Percent",
#         "Granulocyte Count", 
#         "Eosinophil Count", 
#         "Large Platelets",
#         'Immature Granulocytes',
#     ],
#     "BMP":[
#         "Sodium",
#         "Potassium", 
#         "Calcium, Total",
#         "Urea Nitrogen",
#         "Creatinine", 
#         "Bicarbonate", 
#         "Glucose",
#         "Chloride", 
#     ],
#     "Reticulocyte Count": [
#         "Reticulocyte Count, Automated", 
#         "Reticulocyte Count, Absolute", 
#         "Reticulocyte Count, Manual",
#         'Reticulocyte Count',
#     ],
#     "Blood Cell Morphology": [
#         "Platelet Smear", 
#         "RBC Morphology", 
#         "Acanthocytes", 
#         "Basophilic Stippling", 
#         "Target Cells", 
#         "Pappenheimer Bodies", 
#         "Fragmented Cells", 
#         "Plasma Cells", 
#         "Promyelocytes", 
#         "Macrocytes", 
#         "Microcytes", 
#         "Elliptocytes", 
#         "Spherocytes", 
#         "Sickle Cell Preparation", 
#         "Poikilocytosis", 
#         "Schistocytes", 
#         "Teardrop Cells", 
#         "Echinocytes", 
#         "Pencil Cells", 
#         "Howell-Jolly Bodies", 
#         "Immature Granulocytes", 
#         "Anisocytosis", 
#         "Ovalocytes", 
#         "Myelocytes", 
#         "Nucleated Red Cells", 
#         "Atypical Lymphocytes",
#         "Blasts", "Metamyelocytes", 
#         "Polychromasia", "Hypochromia", "Young Cells", 
#         "MacroOvalocytes", "Platelet Clumps", "Bite Cells", 
#         "Other Cells",
#         "Bands", 
#         "Hypersegmented Neutrophils", 
#         "Immature Granulocytes",
#         'Sickle Cells',
#     ],
#     "Liver Function Tests": [
#         "Alanine Aminotransferase (ALT)", 
#         "Asparate Aminotransferase (AST)", 
#         "Alkaline Phosphatase", 
#         "Bilirubin, Direct", 
#         "Bilirubin, Indirect", 
#         "Bilirubin, Total", 
#         "Albumin", 
#         "Total Protein", 
#         "Gamma Glutamyltransferase", 
#         "Hepatitis B Surface Antibody", 
#         "Hepatitis B Surface Antigen", 
#         "Hepatitis B Virus Core Antibody", 
#         "Hepatitis C Virus Antibody", 
#         "Hepatitis A Virus Antibody", 
#         "Hepatitis B Core Antibody, IgM", 
#         "Hepatitis A Virus IgM Antibody", 
#         "Hepatitis C Viral Load", 
#         "Hepatitis B Viral Load", 
#         "Anti-Smooth Muscle Antibody", 
#         "Anti-Mitochondrial Antibody",
#         'Haptoglobin',
#         'Protein, Total',
#         'Globulin',
#         "Ferritin",
#         'Ammonia',
#     ],
#     "Renal Function Tests": [
#         "Magnesium", 
#         "Phosphate", 
#         "Anion Gap", 
#         "Albumin", 
#         "Estimated GFR (MDRD equation)", 
#         "Creatinine, Whole Blood", 
#         "Sodium, Whole Blood", 
#         "Potassium, Whole Blood", 
#         "Calcium, Free", 
#         "Uric Acid", 
#         "Calculated Bicarbonate, Whole Blood", 
#         "Osmolality, Measured",
#         "Sodium",
#         "Potassium", 
#         "Calcium, Total",
#         "Urea Nitrogen",
#         "Creatinine", 
#         "Bicarbonate", 
#         "Glucose",
#         "Chloride", 
#         'Protein, Total',
#         'Cryoglobulin',
#     ],
#     "Thyroid Function Tests": [
#         "Thyroid Stimulating Hormone", 
#         "Thyroxine (T4), Free", 
#         "Triiodothyronine (T3)", 
#         "Thyroxine (T4)", 
#         "Thyroid Peroxidase Antibodies", 
#         "Thyroglobulin", 
#         "Anti-Thyroglobulin Antibodies",
#         "Tissue Transglutaminase Ab, IgA", 
#         "Parathyroid Hormone"
#     ],
#     "Coagulation and Hemostasis": [
#         "INR(PT)", 
#         "PT", 
#         "PTT", 
#         "Fibrinogen, Functional", 
#         "D-Dimer", 
#         "Lupus Anticoagulant", 
#         "Thrombin", 
#         "Factor VIII", 
#         "Factor IX", 
#         "Factor X", 
#         "Factor VII", 
#         "Factor II", 
#         "Protein C, Antigen", 
#         "Protein C, Functional", 
#         "Protein S, Antigen", 
#         "Protein S, Functional", 
#         "Von Willebrand Factor Activity", 
#         "Von Willebrand Factor Antigen", 
#         "Antithrombin", 
#         "Inhibitor Screen",
#         "Factor XI", "Factor V", "Heparin", "Heparin, LMW", 
#         "FMC-7", "RFXLDLM", "HIT-Ab Interpreted Result", 
#         "HIT-Ab Numerical Result", "Fibrin Degradation Products"
#     ],
#     "Infection and Immunology": [
#         "C-Reactive Protein", 
#         "Procalcitonin", 
#         "HIV Screen", 
#         "HIV 1 Viral Load", 
#         "HIV 1 Ab Confirmation", 
#         "HIV 2 Ab Confirmation",  
#         "Epstein-Barr Virus EBNA IgG Ab", 
#         "Epstein-Barr Virus VCA IgG Ab", 
#         "Epstein-Barr Virus IgG Ab Value", 
#         "Epstein-Barr Virus VCA IgM Ab", 
#         "Epstein-Barr Virus Interpretation", 
#         "Lyme C6 Ab", 
#         "Toxoplasma IgG Ab", 
#         "Toxoplasma IgG Ab Value", 
#         "Toxoplasma Interpretation", 
#         "Hepatitis C Virus Antibody", 
#         "Anti-Nuclear Antibody", 
#         "Anti-Nuclear Antibody, Titer", 
#         "Double Stranded DNA", 
#         "Rheumatoid Factor", 
#         "Anti-Neutrophil Cytoplasmic Antibody", 
#         "Anti-DGP (IgA/IgG)", 
#         "Cytomegalovirus IgM", 
#         "Hepatitis A Virus IgM Antibody", 
#         "Hepatitis B Surface Antibody", 
#         "Hepatitis C Viral Load",
#         "Ferritin",
#         'Immature Granulocytes',
#         'ANCA Titer',
#         'HLA-DR',
#         "C3",
#         'C4',
#         'Cryoglobulin',
#     ],
#     "Vitamin and Nutritional Studies": [
#         "Vitamin B12", 
#         "Folate", 
#         "25-OH Vitamin D", 
#         "Iron", 
#         "Iron Binding Capacity, Total", 
#         "Transferrin", 
#         "Ferritin", 
#         "Calculated Free Testosterone", 
#         "Sex Hormone Binding Globulin", 
#         "Testosterone",
#         'Homocysteine',
#     ],
#     "Tumor Markers": [
#         "Prostate Specific Antigen", 
#         "Alpha-Fetoprotein", 
#         "Carcinoembryonic Antigen (CEA)", 
#         "Cancer Antigen 27.29", 
#         "CA-125", 
#         "Beta-2 Microglobulin",
#         'Carcinoembyronic Antigen (CEA)',
#     ],
#     "Electrolyte and Acid-Base Balance": [
#         "pCO2", 
#         "pH", 
#         "pO2", 
#         "Oxygen Saturation", 
#         "Base Excess", 
#         "Calculated Total CO2", 
#         "Required O2", 
#         "Oxygen", 
#         "pCO2", 
#         "pO2", 
#         "Oxygen Saturation", 
#         "Base Excess", 
#         "Calculated Total CO2",
#         "lactate"
#     ],
#     "Drug Monitoring": [
#         "Digoxin", 
#         "Vancomycin", 
#         "Gentamicin", 
#         "Tobramycin", 
#         "Acetaminophen", 
#         "Salicylate", 
#         "Barbiturate Screen", 
#         "Benzodiazepine Screen", 
#         "Tricyclic Antidepressant Screen", 
#         "Phenytoin", 
#         "Phenytoin, Free", 
#         "Phenytoin, Percent Free", 
#         "Lithium", 
#         "Methotrexate", 
#         'tacroFK',  
#         'Cyclosporin',  
#         'Rapamycin', 
#         'Amikacin',  
#         "Valproic Acid"
#     ],
#     "Blood Type and Antibody Screening": [ 
#         "Hemoglobin A2", 
#         "Hemoglobin C", 
#         "Hemogloblin A", 
#         "Hemogloblin S", 
#     ],
#     "Immunoglobulins": [
#         "Immunoglobulin A", 
#         "Immunoglobulin G", 
#         "Immunoglobulin M", 
#         "Free Kappa", 
#         "Free Kappa/Free Lambda Ratio", 
#         "Free Lambda", 
#         "Kappa", 
#         "Lambda",
#         'Protein Electrophoresis',
#     ],
#     "Lipid Panel":[
#         'Cholesterol, Total',
#         'Cholesterol, LDL, Measured',
#         'Cholesterol, HDL',  
#         'Cholesterol Ratio (Total/HDL)',  
#         'Triglycerides',
#         "Cholesterol, LDL, Calculated",
#     ],
#     #免疫表型分析
#     "Immunophenotyping": [
#         "CD20", "CD20 %", "CD3 %", "CD4/CD8 Ratio", "CD3 Absolute Count", 
#         "CD4 Absolute Count",
#         "CD8 Cells, Percent", "CD19", "CD19 %", "CD19 Absolute Count",
#         "CD4 Cells, Percent", "CD3 Cells, Percent", "CD5", 
#         "CD5 %", "CD5 Absolute Count", "CD23", "CD56", "CD16/56%",
#         "CD16/56 Absolute Count", "CD14", "CD13", "CD15", 
#         "CD33", "CD34", "CD45", "CD117", "CD7", "CD2",
#         'Absolute CD4 Count','CD20 Absolute Count','CD64','CD10','Absolute CD8 Count',
#         'Absolute CD3 Count',
#     ],
#     #病毒学
#     "Virology": [
#         "CMV IgG Ab", "CMV IgG Ab Value", "CMV IgM Ab", 
#         "CMV Interpretation", "Epstein-Barr Virus EBNA IgG Ab", 
#         "Epstein-Barr Virus IgM Ab Value", "Lyme G and M Value","Cytomegalovirus Viral Load",
#     ],
#     "HbA1c": [
#         "% Hemoglobin A1c",
#         "eAG",
#     ],
#     "Thrombophilia Panel":[
#         'Anticardiolipin Antibody IgG',
#         'Anticardiolipin Antibody IgM',
#         'Homocysteine', 
#     ]
# }
####################################################### RAG data
# diag = pd.read_csv('../data/heart/heart_diagnoses.csv')
# lab = pd.read_csv('../data/heart/heart_labevents_first_lab.csv')
# micro = pd.read_csv('../data/heart/heart_microbiologyevents_first_micro.csv')

# data_ = []
# for row in diag[3500:].iterrows():
#     print(diag.shape[0])
#     dict_ = {}
#     dict_[row[1]['hadm_id']] = row[1]['HPI']
#     data_.append(dict_.copy())

# with open('./RAG_data/HPI.json', 'w') as f:
#     json.dump(data_, f, indent=4)

# import torch
# import torch.nn as nn
# from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
# from transformers.generation.utils import GenerationConfig
# import json
# import math
# import template
# from tqdm import tqdm
# model_path = "../llms/llama3.1-8b-instruct"
# # model_path = '../llms/Llama-2-7b-chat-hf/'
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
# model.generation_config = GenerationConfig.from_pretrained(model_path)
# model.to("cuda")
# model.eval()

# def generate_response(message):
#     messages = []
#     messages.append({"role": "user", "content": message })
#     prompt = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
#     outputs = model.generate(prompt, max_new_tokens=512, num_return_sequences=1)
#     response_text = tokenizer.decode(outputs[0][len(prompt[0]):], skip_special_tokens=True)
#     return response_text

# data_ = []
# for row in tqdm(diag[3500:].iterrows(), total=(diag.shape[0]-3500)):
#     image_str = ''
#     lab_str = ''
#     HPI_summary = generate_response(template.HPI_summary.format(HPI=row[1]['HPI']))
#     physical_summary = generate_response(template.physicial_summary.format(physical_exam=row[1]['physical_exam']))
#     lab_hadm = lab[lab['hadm_id'] == row[1]['hadm_id']]
#     micro_hadm = micro[micro['hadm_id'] == row[1]['hadm_id']]
#     if not lab_hadm.empty:
#         for row_l in lab_hadm.iterrows():
#             lab_result = str(row_l[1]['fluid'] + ' ' + row_l[1]['label']) + ' ' + str(row_l[1]['valuenum']) + ' ' + str(row_l[1]['valueuom'])
#             if pd.notna(row_l[1]['flag']):
#                 lab_result = lab_result + ' ' + str(row_l[1]['flag'])
#             if pd.notna(row_l[1]['comments']):
#                 lab_result = lab_result + ' ' + str(row_l[1]['comments'])
#             lab_str += (lab_result + '\n') 
#     if not micro_hadm.empty:
#         inter = {"S": 'sensitive', 'R': 'resistant', 'I':'intermediate', 'P':"pending"}
#         for row_m in micro_hadm.iterrows():
#             result = ""
#             if pd.notna(row_m[1]['test_name']):
#                 result += " " + str(row_m[1]['test_name'])
#             if pd.notna(row_m[1]['org_name']):
#                 result += " " + str(row_m[1]['org_name'])
#             if pd.notna(row_m[1]['ab_name']):
#                 result += " " + str(row_m[1]['ab_name'])
#             if pd.notna(row_m[1]['interpretation']):
#                 result += " " + inter[str(row_m[1]['interpretation'])]
#             if pd.notna(row_m[1]['comments']):
#                 result += " " + str(row_m[1]['comments'])
#             if result == "":
#                 result = "NO GROWTH."
#             lab_str += (result + '\n')
#     lab_summary = generate_response(template.lab_summary.format(lab_exam=lab_str))
#     if len(row[1]['X-ray']) > 2:
#         image_str += ("X-ray:\n" + row[1]['X-ray'] + "\n")
#     if len(row[1]['CT']) > 2:
#         image_str += ("CT:\n" + row[1]['CT']+ "\n")
#     if len(row[1]['Ultrasound']) > 2:
#         image_str += ("Ultrasound:\n" + row[1]['Ultrasound']+ "\n")
#     if len(row[1]['CATH']) > 2:
#         image_str += ("CATH:\n" + row[1]['CATH']+ "\n")
#     if len(row[1]['MRI']) > 2:
#         image_str += ("MRI:\n" + row[1]['MRI']+ "\n")
#     if len(row[1]['ECG']) > 2:
#         image_str += ("ECG:\n" + row[1]['ECG']+ row[1]['reports']+ "\n")
#     else:
#         image_str += ("ECG:\n" + row[1]['reports']+ "\n")
#     image_summary = generate_response(template.image_summary.format(image_exam=image_str))
#     summary = generate_response(template.summary.format(HPI=HPI_summary, physical_summary=physical_summary, lab_summary=lab_summary, image_summary=image_summary))
#     print(summary)
#     data_.append({row[1]['hadm_id']: summary})
#     print(data_)

# with open('./RAG_data/RAG_data_summary.json', 'w') as f:
#     json.dump(data_, f)

#######################################################
# df = pd.read_csv('../data/heart/heart_diagnoses_all_true.csv')
# filtered_df = df[df['seq_num'] == 1]
# filtered_df['icd_code_prefix'] = filtered_df['icd_code'].str[:3]
# prefix_counts = filtered_df['icd_code_prefix'].value_counts()
# print(prefix_counts)

# diag = pd.read_csv('../data/heart/heart_diagnoses.csv')
# print(diag.shape[0])
# print(len(diag['X-ray'].unique().tolist()))
# print(len(diag['CT'].unique().tolist()))
# print(len(diag['Ultrasound'].unique().tolist()))
# print(len(diag['CATH'].unique().tolist()))
# print(len(diag['MRI'].unique().tolist()))
# print(len(diag['ECG'].unique().tolist()))

# lab = pd.read_csv('../data/heart/heart_labevents_first_lab.csv')
# micro = pd.read_csv('../data/heart/heart_microbiologyevents_first_micro.csv')

# print(lab[['label','fluid']].drop_duplicates().shape[0])




#######################################################
with open('./result/heart_result_llama70b_0-2000.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pathways = []

for item in data:
    for key, value in item.items():
         pathways.append(value[0]['clinical_pathway'])

from graphviz import Digraph
def create_simplified_pathway_graph(pathways, min_count=50, label_font_size=8):
    dot = Digraph(comment='Simplified Aggregated Clinical Pathways')
    dot.attr(fontsize='10')  # 控制全图字体的大小
    dot.attr('node', fontsize='10')  # 控制节点字体的大小
    dot.attr('edge', fontsize=str(label_font_size))  # 控制边标签字体大小
    # 统计每对节点间的过渡频次
    transitions = {}
    for pathway in pathways:
        for i in range(len(pathway) - 1):
            edge = (pathway[i], pathway[i+1])
            if edge in transitions:
                transitions[edge] += 1
            else:
                transitions[edge] = 1

    # 添加节点和符合条件的边（出现次数 >= min_count）
    for (start, end), count in transitions.items():
        if count >= min_count:
            dot.node(start, start)  # 确保节点在图中
            dot.node(end, end)      # 确保节点在图中
            dot.edge(start, end, label=str(count))  # 仅添加边，且显示出现次数

    # 渲染图形，输出为PDF文件
    dot.render('70b', format='pdf', cleanup=True)

create_simplified_pathway_graph(pathways)

    
        
        