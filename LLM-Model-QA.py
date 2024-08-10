import pdfplumber

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# import openai

# openai.api_key = 'use-api-key'
# def extract_details_with_openai(text):
    
#     response = openai.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant specialized in extracting information from text."},
#             {"role": "user", "content": f"Extract the customer details, product information, and total amount from the following text: {text}"}
#         ],
#         max_tokens=500
#     )

#     extracted_details = response['choices'][0]['message']['content']
#     return extracted_details

from transformers import pipeline

def extract_details_with_transformers(text):
    # Load a pipeline for question-answering
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    # Define questions to extract customer details, product info, and total amount
    questions = [
        {"question": "Who is the customer?", "context": text},
        {"question": "What products are listed?", "context": text},
        {"question": "What is the total amount?", "context": text}
    ]

    # Extract answers for each question
    answers = {}
    for q in questions:
        result = qa_pipeline(q)
        answers[q["question"]] = result['answer']
    
    return answers

text_from_pdf = extract_text_from_pdf("./sample.pdf")
details = extract_details_with_transformers(text_from_pdf)
print('extracted details: ',details)

# from transformers import pipeline

# # Load a pretrained model for text extraction
# nlp = pipeline("ner", model="dslim/bert-base-NER")

# def extract_details_with_huggingface(text):
#     ner_results = nlp(text)
    
#     customer = []
#     product = []
#     total_amount = None

#     for entity in ner_results:
#         if entity['entity'] == 'B-CUSTOMER':
#             customer.append(entity['word'])
#         elif entity['entity'] == 'B-PRODUCT':
#             product.append(entity['word'])
#         elif entity['entity'] == 'B-MONEY':
#             total_amount = entity['word']

#     details = {
#         'customer': ' '.join(customer),
#         'product': ' '.join(product),
#         'total_amount': total_amount
#     }
#     return details

# text_from_pdf = extract_text_from_pdf("./sample.pdf")
# details = extract_details_with_huggingface(text_from_pdf)
# print(details)