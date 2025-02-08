first_temp = """
Now you are a cardiologist. A patient is being admitted for the first time, and his chief complaint is:
{chief_complaint}
The patient's medical history is as follows:
{phi}  
As the patient's attending physician, guide the patient to undergo an in-hospital examination. Each time, you can only request one specific test. Please clearly specify the name of the test.
I will provide you with information according to your request.
The available tests include physical examination, laboratory tests, and imaging tests.
Your answer format:
[
Analysis:
Next recommended examination:the name of medical examination
]
If you believe that the patient no longer needs further examinations, make the final diagnosis.
[
Final diagnosis: List of disease names separated by commas
Treatment:
]
"""

first_temp_RAG = """
Now you are a cardiologist. A patient is being admitted for the first time, and his chief complaint is:
{chief_complaint}
The patient's medical history is as follows:
{phi}  
According to the hospital case database, the examination measures we have taken for similar patients in the past are as follows:
{exam_for_hadm} 
Finally, the patient was diagnosed with:{final_diagnosis}
As the patient's attending physician, guide the patient to undergo an in-hospital examination. Each time, you can only request one specific test. Please clearly specify the name of the test.
I will provide you with information according to your request.
The available tests include physical examination, laboratory tests, and imaging tests.
Note: Normally, we would first conduct a physical examination on the patient.
Your answer format:
[
Analysis:
Next recommended examination:the name of medical examination
]
If you believe that the patient no longer needs further examinations, make the final diagnosis.
[
Final diagnosis: List of disease names separated by commas
Treatment:
]
"""

existed_temp = """
This is the patient's {exam} result：
{content}
Please continue your medical process.
I will provide you with information according to your request.
The available tests include physical examination, laboratory tests, and imaging tests. Each time, you can only request one specific test. Please clearly specify the name of the test.
Your answer format:
[
Analysis:
Next recommended examination:the name of medical examination
]
If you believe that the patient no longer needs further examinations, make the final diagnosis.
[
Final diagnosis: List of disease names separated by commas
Treatment:
]
"""

unexisted_temp = """
There are no {exam} results for the patient.
Please summarize the patient's condition, then continue your medical process.
I will provide you with information according to your request.
The available tests include physical examination, laboratory tests, and imaging tests.Each time, you can only request one specific test. Please clearly specify the name of the test.
Your answer format:
[
Analysis:
Next recommended examination:the name of medical examination
]
If you believe that the patient no longer needs further examinations, make the final diagnosis.
[
Final diagnosis: List of disease names separated by commas
Treatment:
]
"""

existed_temp_RAG = """
This is the patient's {exam} result：
{content}
Please continue your medical process.
I will provide you with information according to your request.
The available tests include physical examination, laboratory tests, and imaging tests. Each time, you can only request one specific test. Please clearly specify the name of the test.
According to the hospital case database, the examination measures we have taken for similar patients in the past are as follows:
{exam_for_hadm}
Your answer format:
[
Analysis:
Next recommended examination:the name of medical examination
]
If you believe that the patient no longer needs further examinations, make the final diagnosis.
[
Final diagnosis: List of disease names separated by commas
Treatment:
]
"""

unexisted_temp_RAG = """
There are no {exam} results for the patient.
Please summarize the patient's condition, then continue your medical process.
I will provide you with information according to your request.
The available tests include physical examination, laboratory tests, and imaging tests.Each time, you can only request one specific test. Please clearly specify the name of the test.
According to the hospital case database, the examination measures we have taken for similar patients in the past are as follows:
{exam_for_hadm}
Your answer format:
[
Analysis:
Next recommended examination:the name of medical examination
]
If you believe that the patient no longer needs further examinations, make the final diagnosis.
[
Final diagnosis: List of disease names separated by commas
Treatment:
]
"""

summary_temp = """
This is the patient's {exam} result：
{content}
Please summarize the known patient's condition.
If you believe that the patient no longer needs further examinations, make the final diagnosis.
Your answer format:
[
Summary:
Final diagnosis: List of disease names separated by commas
Treatment:
]
Alternatively, you can continue to request the patient to undergo other examinations.Each time, you can only request one specific test. Please clearly specify the name of the test.
I will provide you with information according to your request.
The available tests include physical examination, laboratory tests, and imaging tests.
Your answer format:
[
Summary:
Next recommended examination:the name of medical examination
]
"""

been_chosen_temp = """
The {exam} results have been provided to you.
Please continue your medical process.
The available tests include physical examination, laboratory tests, and imaging tests.
I will provide you with information according to your request.
Each time, you can only request one specific test. Please clearly specify the name of the test.
Your answer format:
[
Analysis:
Next recommended examination: the name of medical examination
]
If you believe that the patient no longer needs further examinations, make the final diagnosis.
[
Final diagnosis: List of disease names separated by commas
Treatment:
]
"""

been_chosen_temp_RAG = """
The {exam} results have been provided to you.
Please continue your medical process.
The available tests include physical examination, laboratory tests, and imaging tests.
I will provide you with information according to your request.
Each time, you can only request one specific test. Please clearly specify the name of the test.
According to the hospital case database, the examination measures we have taken for similar patients in the past are as follows:
{exam_for_hadm}
Your answer format:
[
Analysis:
Next recommended examination: the name of medical examination
]
If you believe that the patient no longer needs further examinations, make the final diagnosis.
[
Final diagnosis: List of disease names separated by commas
Treatment:
]
"""

final_temp = """
Please summarize the known patient's condition.
And then make a final diagnosis based on the patient's condition and list the diseases you believe the patient has.
Your answer format:
[
Summary:
Final diagnosis: List of disease names separated by commas
Treatment:
]
"""


HPI_summary = """
This is the chief complaint and HPI of a patient:
{chief_complaint}
{HPI}
Please summarize the known patient's condition in a coherent paragraph.
"""

physicial_summary = """
This is the physical examination result of a patient:
{physical_exam}
Please summarize the known patient's condition in a coherent paragraph.
"""

lab_summary = """
This is the laboratory test result of a patient:
{lab_exam}
Please summarize the known patient's condition in a coherent paragraph.
"""

image_summary = """
This is the image test result of a patient:
{image_exam}
Please summarize the known patient's condition in a coherent paragraph.
"""

micro_summary = """
This is the microorganism test result of a patient:
{micro_exam}
Please summarize the known patient's condition in a coherent paragraph.
"""

summary="""
This is the summary of a patient's present health information, physical examination, laboratory test, imaging test and microorganism test:
{HPI}
{physical_summary}
{image_summary}
{lab_summary}
{micro_summary}
Please summarize the patient's condition in a coherent paragraph.
And then make a final diagnosis based on the patient's condition and list the diseases you believe the patient has.
Your answer format:
[
Summary:
Final diagnosis: List of disease names separated by commas
Treatment:
]
"""

summary_v2="""
This is the summary of a patient's chief complaint, present health information, physical examination, laboratory test, imaging test and microorganism test:
{chief_complaint}
{HPI}
{physical_summary}
{image_summary}
{lab_summary}
{micro_summary}
Please summarize the patient's condition in a coherent paragraph.
And then make a final diagnosis based on the patient's condition and list the diseases you believe the patient has.
Your answer format:
[
Summary:
Final diagnosis: List of disease names separated by commas
Treatment:
]
"""