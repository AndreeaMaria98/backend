from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
import spacy
import re
from dictionaries import abbreviations, specializations

app = Flask(__name__)
nlp = spacy.load("ro_core_news_md")
df = pd.read_csv('set_de_date.csv')

# Function to replace specializations full names with their abbreviations
def replace_specializations_with_short_form(text):
    for full_name, abbreviation in specializations.items():
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
def remove_punctuation(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.pos_ not in ["PUNCT"]:
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
    question = remove_punctuation(question)
    question = replace_abbreviations(question)
    question = replace_specializations_with_short_form(question)
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
@cross_origin(origin='localhost', headers=['Content- Type'])
def home():
    if request.method == 'POST':
        question = request.get_json()['question']
        link, etiquette = search_response(question, df)
        if link == None:
            return jsonify({'response': "Îmi pare rău, dar nu am gasit informațiile pe care le-ai solicitat."})
        link = "http://www.ace.ucv.ro" + link
        return jsonify({'response': "Informațiile pe care le cauți pot fi găsite la următorul link: {}".format(link)})
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)