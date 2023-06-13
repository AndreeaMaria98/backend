from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import spacy
import re
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
nlp = spacy.load("ro_core_news_md")
df = pd.read_csv('set_de_date.csv')

abbreviations = {
    "ang": "angajati",
    "an": "anul",
    "sem": "semestrul",
    "spec": "specializare",
    "pt": "pentru",
    "dc": "de ce" 
}

inverse_abbreviations = {    
    "ingineria sistemelor multimedia": "ISM",
    "automatica si informatica aplicata": "AIA",
    "calculatoare romana" : "CR",
    "calculatoare engleza" : "CE",
    "electronica aplicata" : "ELA",
    "mecatronica" : "MCT",
    "robotica" : "ROB",
    "sisteme automate incorporate" : "SAI",
    "tehnologii informatice in ingineria sistemelor" : "TIS",
    "tehnologii informatice in ingineria sistemelor" : "TIIS",
    "information systems for e-business" : "ISB",
    "ingineria calculatoarelor si comunicatiilor" : "ICC",
    "inginerie software" : "IS",
    "sisteme de conducere in robotica" : "SCR"}

# Function to replace full names in text
def replace_full_names_with_abbreviations(text):
    for full_name, abbreviation in inverse_abbreviations.items():
        # Search for full names regardless of the case (uppercase/lowercase) in which they are written
        pattern = re.compile(re.escape(full_name), re.IGNORECASE)
        # Replace all the matches with the corresponding abbreviation
        text = pattern.sub(abbreviation, text)
    return text

# Function to replace abbreviations in text
def replace_abbreviations(text):
    words = text.split()
    for i in range(len(words)):
        if words[i].lower() in abbreviations:
            words[i] = abbreviations[words[i].lower()]
    return " ".join(words)

# Function to extract keywords from the question
def extract_keywords(question):
    doc = nlp(question)
    keywords = []
    for token in doc:
        keywords.append(token.text.lower())
    return keywords

# Function to remove the punctuation marks and auxiliary verbs
def remove_punct_and_aux(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.pos_ not in ["PUNCT", "AUX"]:
            tokens.append(token.text)
    return " ".join(tokens)

# Process the questions from the dataset and store them in a new column
df['processed_question'] = df['intrebare'].apply(lambda x: nlp(x))

def compute_similarity(ds_processed_question, user_question):
    similarity = ds_processed_question.similarity(user_question)

    # Create a bonus if the keywords are in the same order
    # Extract the words from database processed question and user input question
    words_x = [token.text for token in ds_processed_question]
    words_question = [token.text for token in user_question]
    
    # Check if the words in question appear in the same order in the database processed question
    order_bonus = 0
    i = 0
    for word in words_question:
        if word in words_x[i:]:
            order_bonus += 0.1  # choose a small bonus because we don't want it to be the dominant factor in the similarity calculation
            i = words_x.index(word)
    
    # Add the bonus to the similarity score
    similarity += order_bonus
    
    return similarity

def search_response(question, df):
    question = remove_punct_and_aux(question)
    question = replace_full_names_with_abbreviations(question)
    question = replace_abbreviations(question)
    question = nlp(question)
    
    # Calculate the similarity using the processed questions from the data set
    df['similarity'] = df['processed_question'].apply(lambda x: compute_similarity(x, question))

    df = df.sort_values(by='similarity', ascending=False)

    if df['similarity'].iloc[0] > 0.7:
        link = df['link'].iloc[0]
        eticheta = df['eticheta'].iloc[0]
        return link, eticheta
    else:
        return None, None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        intrebare = request.form.get('question')
        # aici apelezi funcția ta pentru a căuta răspunsul
        link, eticheta = search_response(intrebare, df)
        # aici ar trebui să returnezi răspunsul într-un fel care să aibă sens pentru aplicația ta
        # de exemplu, poate dorești să încorporezi răspunsul într-un alt șablon HTML și să îl returnezi
        return "Link: {} Eticheta: {}".format(link, eticheta)
    else:
        # în cazul unei cereri GET, doar returnează șablonul
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)