from google.cloud import language_v1
from flask import Flask
from flask_cors import CORS
import wikipediaapi
from dotenv import load_dotenv
import json
import re
import os

### activare env: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -> licenta\Scripts\Activate

#Incarcare .env
load_dotenv()

app = Flask("DialogInpainting")
CORS(app)

# Pas 1: Extragere continut articol de pe Wikipedia
def get_wikipedia_content(page_title):
    wiki_wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="dialog inpainting")

    page = wiki_wiki.page(page_title)
    if page.exists():
        content = page.text
    else:
        print(f"The page '{page_title}' does not exist on Wikipedia.")
    return content


# Pas 2: Parsare paragrafe in propozitii folosin Google Cloud Natural Language API
def split_into_sentences(content, api_key_path):
    with open(api_key_path, "r") as json_file:
        api_key_data = json.load(json_file)

    client = language_v1.LanguageServiceClient.from_service_account_info(
        api_key_data)

    document = language_v1.Document(
        content=content, type_=language_v1.Document.Type.PLAIN_TEXT)

    # Enable syntax analysis for sentence splitting
    features = {"extract_syntax": True, "extract_entities": False,
                "extract_document_sentiment": False, "extract_entity_sentiment": False, "classify_text": False}

    response = client.annotate_text(document=document, features=features)

    sentences = [sentence.text.content.strip() for sentence in response.sentences if re.match(
        r'^[A-Z].*\.$', sentence.text.content)]

    return sentences


# Pas 3: Afisarea primelor 6 prop din fiecare paragraf
def print_first_six_sentences(sentences):
    # for paragraph in sentences:
    #     paragraph_sentences = paragraph.split('\n')[:6]
    #     print("\n".join(paragraph_sentences))
    #     # print("\n")
    for paragraph in sentences:
        paragraph_sentences = paragraph.split('\n')
        num_sentences = min(6, len(paragraph_sentences))
        print("\n".join(paragraph_sentences[:num_sentences]))
        # If the paragraph has fewer than six sentences, print all sentences
        if num_sentences < 6:
            print("\n".join(paragraph_sentences[num_sentences:]))
        print("\n")


if __name__ == "__main__":
    article_title = input("Enter the Wikipedia article title: ")
    wikipedia_content = get_wikipedia_content(article_title)

    sentences = split_into_sentences(
        wikipedia_content, api_key_path=os.getenv("API_KEY_PATH"))
    print_first_six_sentences(sentences)
