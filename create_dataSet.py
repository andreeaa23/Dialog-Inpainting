import json
import os
from google.cloud import language_v1

#labels: text, title, aid, bid, id
input_file_path = 'OR_QuAC.txt'
output_file_path = 'Wiki_Dialog.json'

def split_paragraphs_into_sentences(paragraph, client):

    document = language_v1.Document(content=paragraph, type_=language_v1.Document.Type.PLAIN_TEXT)
    features = {"extract_syntax": True, "extract_entities": False,
                "extract_document_sentiment": False, "extract_entity_sentiment": False, "classify_text": False}
    response = client.annotate_text(document=document, features=features)
    
    sentences = [sentence.text.content for sentence in response.sentences]
    return sentences


def extract_fields_to_new_json(input_file_path, output_file_path, limit=1000):
    with open("cheie2.json", "r") as json_file:  
        api_key_data = json.load(json_file)

    client = language_v1.LanguageServiceClient.from_service_account_info(api_key_data)

    new_data = []
    with open(input_file_path, 'r') as input_file:
        for i, line in enumerate(input_file):
            if i >= limit:  # Break the loop after processing 'limit' lines
                break
            item = json.loads(line)
            sentences = split_paragraphs_into_sentences(item['text'], client)
            new_row = {
                'pid': item['id'],
                'title': item['title'],
                'passage': item['text'],
                'sentences': sentences
                #'utterances': cand o sa fac algoritmul, prima prop o sa fie: "Hi, I'm your automated assistant. I can
# answer your questions about Mother Mary Alphonsa."
                #author_num": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] #0-server, 1-user
            }
            new_data.append(new_row)

    with open(output_file_path, 'w') as output_file:
        json.dump(new_data, output_file, indent=2)

    print(f"New dataset created with {len(new_data)} entries.")
    
    
def print_first_n_rows(file_path, n=5):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

            for i in range(min(n, len(data))):
                print(json.dumps(data[i], indent=2))
    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except json.JSONDecodeError:
        print("File is not a valid JSON. Please check the file content.")
    

    
extract_fields_to_new_json(input_file_path, output_file_path, 1000)
print_first_n_rows(output_file_path)
