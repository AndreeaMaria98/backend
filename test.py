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

def replace_full_names_with_abbreviations(text):
    for full_name, abbreviation in inverse_abbreviations.items():
        # Căutăm numele complete indiferent de cazul (majuscule/minuscule) în care sunt scrise
        pattern = re.compile(re.escape(full_name), re.IGNORECASE)
        # Înlocuim toate potrivirile cu prescurtarea corespunzătoare
        text = pattern.sub(abbreviation, text)
    return text

# Funcție pentru înlocuirea prescurtărilor din text
def replace_abbreviations(text):
    words = text.split()
    for i in range(len(words)):
        if words[i].lower() in abbreviations:
            words[i] = abbreviations[words[i].lower()]
    return " ".join(words)

# Funcție pentru extragerea cuvintelor cheie din întrebare
def extract_keywords(question):
    doc = nlp(question)
    keywords = []
    for token in doc:
        keywords.append(token.text.lower())
    return keywords

def remove_punctuation_and_pos(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.pos_ not in ["PUNCT", "AUX"]:
            tokens.append(token.text)
    return " ".join(tokens)

# Procesăm întrebările din setul de date și le stocăm într-o coloană nouă
df['processed_question'] = df['intrebare'].apply(lambda x: nlp(x))

def compute_similarity(x, question):
    # return x.similarity(question)
    similarity = x.similarity(question)

    # Creăm un bonus dacă cuvintele cheie sunt în aceeași ordine
    # Extragem cuvintele din x și question
    words_x = [token.text for token in x]
    words_question = [token.text for token in question]
    
    # Verificăm dacă cuvintele din question apar în aceeași ordine în x
    order_bonus = 0
    i = 0
    for word in words_question:
        if word in words_x[i:]:
            order_bonus += 0.1  # alegem un bonus mic, deoarece nu dorim să fie factorul dominant în calculul similarității
            i = words_x.index(word)
    
    # Adăugăm bonusul la scorul de similaritate
    similarity += order_bonus
    
    return similarity

def cauta_raspuns(intrebare, df):
    intrebare = remove_punctuation_and_pos(intrebare)
    intrebare = replace_full_names_with_abbreviations(intrebare)
    print(intrebare)
    intrebare = replace_abbreviations(intrebare)
    question = nlp(intrebare)
    
    # Calculăm similaritatea folosind întrebările procesate din setul de date
    df['similarity'] = df['processed_question'].apply(lambda x: compute_similarity(x, question))

    df = df.sort_values(by='similarity', ascending=False)
    # print(df)

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
        link, eticheta = cauta_raspuns(intrebare, df)
        # aici ar trebui să returnezi răspunsul într-un fel care să aibă sens pentru aplicația ta
        # de exemplu, poate dorești să încorporezi răspunsul într-un alt șablon HTML și să îl returnezi
        return "Link: {} Eticheta: {}".format(link, eticheta)
    else:
        # în cazul unei cereri GET, doar returnează șablonul
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)