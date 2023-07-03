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
        pattern = re.compile(r'\b' + re.escape(full_name) + r'\b', re.IGNORECASE)
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

# Function to remove the punctuation marks and auxiliary verbs
def remove_punctuation(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.pos_ not in ["PUNCT"]:
            tokens.append(token.text)
    return " ".join(tokens)

# Process the questions from the dataset and store them in a new column
df['processed_question'] = df['intrebare'].apply(lambda x: nlp(x.lower()))

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

# Function to calculate the number of unmatched words
def compute_unmatched_words(user_question, ds_processed_question):
    user_words = set([token.text.lower() for token in user_question])
    ds_words = set([token.text.lower() for token in ds_processed_question])

    # Find the words that are in user_words but not in ds_words
    unmatched_words = ds_words.difference(user_words)

    return len(unmatched_words)

def search_response(question, df):
    question = question.lower()
    question = remove_punctuation(question)
    question = replace_abbreviations(question)
    question = replace_specializations_with_short_form(question)
    question = nlp(question)
    
    # Calculate the similarity using the processed questions from the data set
    df['similarity'] = df['processed_question'].apply(lambda x: compute_similarity(x, question))
    
    # Subtract a penalty based on the number of unmatched words
    df['similarity'] = df.apply(lambda row: row['similarity'] - 0.01 * compute_unmatched_words(question, row['processed_question']), axis=1)

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