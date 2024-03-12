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
# CORS(app, resources={r"/api/*": {"origins": "*"}})

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
    titles = request.json.get('titles') # Assume 'titles' is a list of titles
    
    if not titles:
        return jsonify({"message": "No titles provided"}), 400
    
    # Find the user document using the current_user ID and update it
    result = collection.update_one(
        {"_id": ObjectId(current_user)},{"$addToSet": {"searched_titles": {"$each": titles}}},  # Use $addToSet with $each to add titles without duplication
        upsert=True  # If the document doesn't exist, create it
    )

    if result.modified_count > 0:
        return jsonify({"message": "Titles added successfully"}), 200
    else:
        return jsonify({"message": "Failed to add titles"}), 500
    
@app.route('/getTitles', methods=['GET'])
@jwt_required()
def get_titles():
    current_user = get_jwt_identity()
    
    # Find the user document using the current_user ID
    user = collection.find_one({"_id": ObjectId(current_user)})
    
    # Check if the user document has the 'searched_titles' field
    if user and 'searched_titles' in user:
        return jsonify({"titles": user['searched_titles']}), 200
    elif user:
        # The user exists but has no titles added yet
        return jsonify({"titles": []}), 200
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
        print(f"The page '{page_title}' does not exist on Wikipedia!")
        
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


@app.route('/getSummary', methods=['GET'])
@jwt_required()
def get_wikipedia_summary(): #ar merge o sumarizare extractiva aici
    # Pas 1: Extragere continut articol de pe Wikipedia
    page_title = request.args.get('page_title')

    wiki_wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="dialog inpainting")

    page = wiki_wiki.page(page_title)
    if page.exists():
        content = page.text
    else:
        print(f"The page '{page_title}' does not exist on Wikipedia!")
        
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
    sentence_count = 0
    
    # Pas 3: Alegerea primelor 6 prop din fiecare paragraf
    for sentence in sentences:
        if sentence_count < 10:
            result.append(sentence)
            sentence_count += 1
        else:
            break  # Stop the loop if we've added 10 sentences
    
    return jsonify(result) 

@app.route('/deleteTitle', methods=['POST'])
@jwt_required()
def delete_title():
    current_user_id = get_jwt_identity()
    data = request.get_json()

    title_to_delete = data.get('title')
    if not title_to_delete:
        return jsonify({"error": "Title is required"}), 400

    # Assuming your user's document contains a field 'searched_titles' which is a list of titles
    result = collection.update_one(
        {"_id": ObjectId(current_user_id)},
        {"$pull": {"searched_titles": title_to_delete}}  # $pull operator removes from an existing array all instances of a value or values that match a specified condition
    )

    if result.modified_count > 0:
        return jsonify({"message": "Title deleted successfully"}), 200
    else:
        # This could happen if the title wasn't in the user's list or the user document couldn't be found
        return jsonify({"error": "Failed to delete title"}), 500


