spacy
flask
pymongo
wikipedia
wikipediaapi
nltk

python -m spacy download en_core_web_sm -> language model for EN



In summary, wikipedia provides a higher-level, more user-friendly interface for quickly fetching information from Wikipedia,
while wikipediaapi offers a lower-level, more customizable way to interact with the Wikipedia API and retrieve specific data.
The choice of which library to use depends on your specific requirements and preferences for accessing Wikipedia data.

doc.sents:

doc.sents is an attribute provided by spaCy that allows you to access the sentences detected in the processed text (doc).
It returns an iterator over sentence spans, where each span represents a sentence in the processed text.
for sent in doc.sents:

This is a for loop that iterates over each sentence span (sent) in the doc.sents iterator.
In each iteration, sent represents a sentence span.
sent.text:

sent is a sentence span, and sent.text retrieves the actual text of the sentence from the span.
List Comprehension:

List comprehension is a concise way to create a new list by applying an expression to each item in an existing iterable (in this case, the sentences in doc.sents).
[sent.text for sent in doc.sents]:

This list comprehension iterates over each sentence span (sent) in doc.sents.
For each sentence span, it retrieves the text of the sentence using sent.text and adds it to the new list being created.
As a result, this list comprehension creates a list (sentences) containing the text of each sentence in the processed text.
In summary, the line sentences = [sent.text for sent in doc.sents] efficiently extracts the text of each sentence from the processed text using spaCy's sentence spans and stores them in a list called sentences. Each element in the list corresponds to a sentence in the text


Deci:
-sa fac rezumat pentru text(ideal ar fi pentru fiecare paragraf in parte)
-apoi sa generez intrebarile si raspunsurile
