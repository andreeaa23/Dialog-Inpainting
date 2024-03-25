from flask_jwt_extended import create_access_token, JWTManager, jwt_required, get_jwt_identity
from google.cloud import language_v1
from flask import request, jsonify, Flask
from nltk import tokenize, word_tokenize
from pymongo import ReturnDocument
from pymongo import MongoClient
from dotenv import load_dotenv
from collections import Counter
from passlib.hash import bcrypt
from datetime import timedelta
from flask_cors import CORS
from bson import ObjectId
import wikipediaapi
import nltk
import spacy
import json
import torch
import re
import os

### activare env: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -> licenta\Scripts\Activate

#Incarcare .env
load_dotenv()
nlp = spacy.load("en_core_web_md") #English model language

app = Flask("DialogInpainting")
CORS(app)

# Conectare MongoDB
#client = MongoClient(os.getenv("MONGO_URI"))
client = MongoClient("mongodb+srv://andreea23:admin@licenta.dulxhyq.mongodb.net/?retryWrites=true&w=majority")
print(client)
db = client[os.getenv("DB_NAME")]
collection = db.get_collection(os.getenv("DB_COLLECTION_NAME"))

# Setari JWT
jwt = JWTManager(app)
secret_key = os.getenv("SECRET_KEY")
app.config['JWT_SECRET_KEY'] = secret_key

try:
    server_info = client.server_info()
    print("Connected to MongoDB, version: ", server_info['version'])
except Exception as e:
    print("Could not connect to MongoDB: %s" % e)
    
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')

    if collection.find_one({"username": username}):
        return jsonify({"message": "Numele de utilizator există deja!"}), 400
    elif collection.find_one({"email": email}):
        return jsonify({"message": "Email-ul există deja!"}), 400
    else:
        hashed_password = bcrypt.hash(password)
        collection.insert_one({"email": email, "username": username, "password": hashed_password})
        return jsonify({"message": "Registration successful"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    user = collection.find_one({"username": username})

    if not user:
        return jsonify({"message": "Invalid username or password!"}), 401

    if bcrypt.verify(password.encode('utf-8'), user["password"]):
        access_token = create_access_token(identity=str(user["_id"]), expires_delta=timedelta(days=7)) # ObjectId isn't JSON serializable
        return jsonify({"message": "Login successful", "access_token": access_token}), 200
    else:
        return jsonify({"message": "Invalid username or password!"}), 401
    
@app.route('/addTitle', methods=['POST'])
@jwt_required()
def add_searched_title():
    current_user = get_jwt_identity()
    titles = request.json.get('titles')
    
    if not titles:
        return jsonify({"message": "No titles provided"}), 400

    updated_document = collection.find_one_and_update(
        {"_id": ObjectId(current_user)},
        {"$addToSet": {"searched_titles": {"$each": titles}}},
        upsert=True,
        return_document=ReturnDocument.AFTER
    )

    if any(title in updated_document['searched_titles'] for title in titles):
        return jsonify({"message": "Titles added or already existed"}), 200
    else:
        return jsonify({"message": "Failed to add titles"}), 500
    
@app.route('/getTitles', methods=['GET'])
@jwt_required()
def get_titles():
    current_user = get_jwt_identity()
    
    # Find the user document using the current_user ID
    user = collection.find_one({"_id": ObjectId(current_user)})
    
    if user and 'searched_titles' in user:
        titles_with_contents = [
            {"title": entry["title"], "summary": entry["summary"]} 
            for entry in user['searched_titles']
        ]
        return jsonify({"searched_titles": titles_with_contents}), 200
    elif user:
        # The user exists but has no titles added yet
        return jsonify({"searched_titles": []}), 200
    else:
        # User document not found
        return jsonify({"message": "User not found"}), 404
    
@app.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    current_user = get_jwt_identity() # userul curent

    new_password = request.json.get('password')
    user = collection.find_one({"_id": ObjectId(current_user)})
    
    if not user:
        return jsonify({'error': 'The user does not exist!'}), 404

    # Schimbare parola 
    user['password'] = bcrypt.hash(new_password)

    # Salvarea in BD
    collection.update_one({"_id": ObjectId(current_user)}, {'$set': user})
    
    return jsonify({'message': 'Password changed successfully!'})

@app.route('/getContent', methods=['GET'])
@jwt_required()
def get_wikipedia_content():
    # Pas 1: Extragere continut articol de pe Wikipedia
    page_title = request.args.get('page_title')

    wiki_wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="dialog inpainting")

    page = wiki_wiki.page(page_title)
    if page.exists():
        content = page.text
    else:
        #print(f"The page '{page_title}' does not exist on Wikipedia!")
        return jsonify({"message": "The page does not exist on English Wikipedia."}), 404
    
    # Pas 2: Parsare paragrafe in propozitii folosin Google Cloud Natural Language API
    with open(os.getenv("API_KEY_PATH"), "r") as json_file:
        api_key_data = json.load(json_file)

    client = language_v1.LanguageServiceClient.from_service_account_info(
    api_key_data)

    document = language_v1.Document(
    content=content, type_=language_v1.Document.Type.PLAIN_TEXT)

    features = {"extract_syntax": True, "extract_entities": False,
                "extract_document_sentiment": False, "extract_entity_sentiment": False, "classify_text": False}

    response = client.annotate_text(document=document, features=features)

    #sentences = [sentence.text.content.strip() for sentence in response.sentences if re.match(r'^[A-Z].*\.$', sentence.text.content)]
    sentences = [sentence.text.content.strip() for sentence in response.sentences if re.match(r'^[A-Z].*\s[A-Za-z0-9]{2,}\.$', sentence.text.content)]

    result = []
    
    # Pas 3: Alegerea primelor 6 prop din fiecare paragraf
    for paragraph in sentences:
        paragraph_sentences = paragraph.split('\n')
        num_sentences = min(6, len(paragraph_sentences))

        # If the paragraph has fewer than six sentences, print all sentences
        if num_sentences < 6:
            result.append("\n".join(paragraph_sentences[:num_sentences]))
    
    return jsonify(result) #return resul of an array of phrases

def make_summarization(text):
    doc = nlp(text)

    STOP_WORDS = set(text.split())
    word_weights={}

    for entity in doc.ents:
        entity_text = entity.text.lower()
        if entity_text in word_weights:
            word_weights[entity_text] += 1
        else:
            word_weights[entity_text] = 1

    for word in word_tokenize(text):
        word = word.lower()
        
        if len(word) > 1 and word not in STOP_WORDS:
            if word in word_weights.keys():            
                word_weights[word] += 1
            else:
                word_weights[word] = 1

    sentence_weights={}
    sentences = tokenize.sent_tokenize(text)
    nr_sent = int(len(sentences) / 3)
    
    for sent in sentences:
        sentence_weights[sent] = 0
        sent_words = word_tokenize(sent)
        sent_entities = [ent.text.lower() for ent in doc.ents]
        for word in sent_words:
            word = word.lower()
            if word in word_weights:
                sentence_weights[sent] += word_weights[word]
            if word in sent_entities:
                sentence_weights[sent] += 1
    
    highest_weights = sorted(sentence_weights.values())[-nr_sent:]
    summary = ""
    
    for sentence, strength in sentence_weights.items():  
        if strength in highest_weights:
            summary += sentence + " "
            
    summary = summary.replace('_', ' ').strip()
    
    return summary

@app.route('/getSummary', methods=['GET'])
@jwt_required()
def get_wikipedia_summary(): 
    page_title = request.args.get('page_title')

    wiki_wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="dialog inpainting")

    page = wiki_wiki.page(page_title)
    if page.exists():
        content = page.text
        summary = make_summarization(content)
    else:
       return jsonify({"message": "The page does not exist on English Wikipedia."}), 404
        
    with open(os.getenv("API_KEY_PATH"), "r") as json_file:
        api_key_data = json.load(json_file)

    client = language_v1.LanguageServiceClient.from_service_account_info(api_key_data)
    document = language_v1.Document(content=summary, type_=language_v1.Document.Type.PLAIN_TEXT)
    features = {"extract_syntax": True, "extract_entities": False,
                "extract_document_sentiment": False, "extract_entity_sentiment": False, "classify_text": False}

    response = client.annotate_text(document=document, features=features)
    sentences = [sentence.text.content.strip() for sentence in response.sentences if re.match(r'^[A-Z].*\s[A-Za-z0-9]{2,}\.$', sentence.text.content)]
        
    result = []
    current_chunk = []
    
    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) == 3:
            result.append(' '.join(current_chunk))
            current_chunk = []
            result.append('\n') # add \n after 3 sentences

    if current_chunk:
        result.append(' '.join(current_chunk))

    final_result = ''.join(result)
    
    #add in the mongo the title with its content
    current_user = get_jwt_identity()
    updated_document = collection.find_one_and_update(
        {"_id": ObjectId(current_user)},
        {"$addToSet": {"searched_titles": {"title": page_title, "summary": final_result}}},
        upsert=True,
        return_document=ReturnDocument.AFTER
    )

    return jsonify(final_result) 

@app.route('/deleteTitle', methods=['POST'])
@jwt_required()
def delete_title():
    current_user_id = get_jwt_identity()
    data = request.get_json()

    title_to_delete = data.get('title')
    if not title_to_delete:
        return jsonify({"error": "Title is required"}), 400

    result = collection.update_one(
            {"_id": ObjectId(current_user_id)},
            {"$pull": {"searched_titles": {"title": title_to_delete}}}
        )


    if result.modified_count > 0:
        return jsonify({"message": "Title deleted successfully"}), 200
    else:
        return jsonify({"error": "Failed to delete title"}), 500


