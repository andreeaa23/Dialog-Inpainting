import wikipediaapi
import spacy
import nltk, nltk.data
from nltk import tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


nlp = spacy.load("en_core_web_sm") #English model language
#nltk.download('punkt')

def extractContentAsSentences(pageTitle):
    try:
        # Create a Wikipedia API object
        wiki_wiki = wikipediaapi.Wikipedia(
            language = "en",
            user_agent = "dialog inpainting")
        
        page = wiki_wiki.page(pageTitle)

        content_copy = ""
        if page.exists():
            content = page.text

            lines = content.split('\n') 
            for line in lines:
                if "See also" in line:
                    break  
                if line.strip().endswith('.') and line.strip()[0].isupper():
                    content_copy += line + '\n'
        else:
            print(f"The page '{pageTitle}' does not exist on Wikipedia.")
        
        return content_copy
    
    except Exception as e:
        return f"An error occurred: {str(e)}"
        
def extractiveSummarization(sentences): #extractive summarization = identif prop importante si extragerea lor in forma originala
    
    num_sentences = int(len(sentences) / 3)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Compute TF-IDF scores for each sentence
    tfidf_matrix = vectorizer.fit_transform(sentences)
    tfidf_scores = tfidf_matrix.toarray().sum(axis=1)

    # Get the indices of the top sentences based on TF-IDF scores
    top_sentence_indices = tfidf_scores.argsort(axis=0)[-num_sentences:]
    top_sentence_indices = top_sentence_indices.flatten()
    
 # Convert indices to integers
    top_sentence_indices = [int(index) for index in top_sentence_indices]
    
    # Sort the indices and select the top sentences
    top_sentence_indices = sorted(top_sentence_indices)
    
    # Ensure the first sentence is always in the summary
    if 0 not in top_sentence_indices:
        top_sentence_indices.insert(0, 0)
    if 1 not in top_sentence_indices:
        top_sentence_indices.insert(1, 1)
        
        
    top_sentences = [sentences[i] for i in top_sentence_indices]

    return ' '.join(top_sentences)
    
    
def tokenize_sentences(sentences):
    tokens = sentences.split(".\n")
    return tokens

def sumarizare(text):
    doc = nlp(text)

    STOP_WORDS = set(text.split())
    word_weights={}

    for ent in doc.ents:
        ent_text = ent.text.lower()
        if ent_text in word_weights:
            word_weights[ent_text] += 1
        else:
            word_weights[ent_text] = 1

    for word in word_tokenize(text):
        word = word.lower()
        if len(word) > 1 and word not in STOP_WORDS:
            if word in word_weights.keys():            
                word_weights[word] += 1
            else:
                word_weights[word] = 1

    sentence_weights={}
    sentences = tokenize.sent_tokenize(text)
    no_sentences = int(len(sentences)/2)
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
    
    highest_weights = sorted(sentence_weights.values())[-no_sentences:]

    summary=""
    for sentence,strength in sentence_weights.items():  
        if strength in highest_weights:
            summary += sentence + " "
    summary = summary.replace('_', ' ').strip()
    
    return summary

if __name__ == "__main__":
    articleTitle = input("Enter the Wikipedia article title: ")
    print("\n")
    sentences = extractContentAsSentences(articleTitle)

    # for sent in sentences:
    #print(sentences)
    #tokens = tokenize_sentences(sentences)
   # print(tokens)
   
    #summary = extractiveSummarization(tokens)
    summary = sumarizare(sentences)
    print(summary)
        


