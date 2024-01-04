from flask_jwt_extended import create_access_token, JWTManager, jwt_required, get_jwt_identity
from google.cloud import language_v1
from flask import request, jsonify, Flask
from flask_cors import CORS
import wikipediaapi
from dotenv import load_dotenv
import json
import re
import os
from pymongo import MongoClient
from passlib.hash import bcrypt
from datetime import timedelta
from bson import ObjectId

### activare env: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -> licenta\Scripts\Activate

#Incarcare .env
load_dotenv()

app = Flask("DialogInpainting")
CORS(app)

# Conectare MongoDB
client = MongoClient(os.getenv("MONGO_URI"))
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
    # Preluare date din request
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
        return jsonify({"message": "Nume utilizator invalid sau parolă invalidă!"}), 401

    if bcrypt.verify(password.encode('utf-8'), user["password"]):
        access_token = create_access_token(identity=str(user["_id"]), expires_delta=timedelta(days=7)) # ObjectId isn't JSON serializable
        return jsonify({"message": "Login successful", "access_token": access_token}), 200
    else:
        return jsonify({"message": "Nume utilizator invalid sau parolă invalidă!"}), 401
    
@app.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    current_user = get_jwt_identity()

    new_password = request.json.get('password')
    user = collection.find_one({"_id": ObjectId(current_user)})
    
    if not user:
        return jsonify({'error': 'Utilizatorul nu există.'}), 404

    # Schimbare parola 
    user['password'] = bcrypt.hash(new_password)

    # Salvarea in BD
    collection.update_one({"_id": ObjectId(current_user)}, {'$set': user})
    
    return jsonify({'message': 'Parola a fost schimbată cu succes!'})

@app.route('/getContent', methods=['GET'])
@jwt_required()
# Pas 1: Extragere continut articol de pe Wikipedia
def get_wikipedia_content():
    data = request.get_json()
    page_title = data.get('page_title')
    
    wiki_wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="dialog inpainting")

    page = wiki_wiki.page(page_title)
    if page.exists():
        content = page.text
    else:
        print(f"The page '{page_title}' does not exist on Wikipedia.")
        
            
    with open(os.getenv("API_KEY_PATH"), "r") as json_file:
        api_key_data = json.load(json_file)

    client = language_v1.LanguageServiceClient.from_service_account_info(
    api_key_data)

    document = language_v1.Document(
    content=content, type_=language_v1.Document.Type.PLAIN_TEXT)

    # Enable syntax analysis for sentence splitting
    features = {"extract_syntax": True, "extract_entities": False,
                "extract_document_sentiment": False, "extract_entity_sentiment": False, "classify_text": False}

    response = client.annotate_text(document=document, features=features)

    #sentences = [sentence.text.content.strip() for sentence in response.sentences if re.match(r'^[A-Z].*\.$', sentence.text.content)]
    sentences = [sentence.text.content.strip() for sentence in response.sentences if re.match(r'^[A-Z].*\s[A-Za-z0-9]{2,}\.$', sentence.text.content)]


    result = []
    
    for paragraph in sentences:
        paragraph_sentences = paragraph.split('\n')
        num_sentences = min(6, len(paragraph_sentences))

        # If the paragraph has fewer than six sentences, print all sentences
        if num_sentences < 6:
            result.append("\n".join(paragraph_sentences[:num_sentences]))
    
    return jsonify(result)

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
