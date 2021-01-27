# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:27:23 2020

@author: Hemakshi Pandey
"""


from flask import Flask, request
from flasgger import Swagger
import nltk
# NLTK is a leading platform for building Python programs to work with human language data.
import pickle 
#Comes handy to save complicated data.Python pickle module is used for serializing and de-serializing python object structures.
import re
#This module provides regular expression matching operations
from nltk.corpus import stopwords
nltk.download('stopwords')
# One of the major forms of pre-processing is to filter out useless data. 
#In natural language processing, useless words (data), are referred to as stop words.
nltk.download('wordnet')
wnlem = nltk.WordNetLemmatizer()
#Lemmatization, unlike Stemming, reduces the inflected words properly ensuring that the root word belongs to the language.
nltk.download('punkt')
#This tokenizer divides a text into a list of sentences, by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences.


# unpickling the classifier 
with open('C:/DEPLOYMENT/Vote_Ensemble.pkl','rb') as Vote_Ensemble_Model:
    classifier = pickle.load(Vote_Ensemble_Model)
    
with open('C:/DEPLOYMENT/Vote_bagofwordsmodel.pkl','rb') as Vote_BOW:
    cv = pickle.load(Vote_BOW)

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict', methods=["GET"])
def Analyse_Section_IT_ACT():
    """Example endpoint returning a prediction of IT ACT SECTION
    ---
    parameters:
      - name: IT_Section_Text
        in: query
        type: string
        required: true
    definitions:
        value:
            type: object
            properties:
                value_name:
                    type: string
                    items:
                        $ref: '#/definitions/Color'
        Color:
            type: string
    responses:
        200:
            description: OK
            schema:
                $ref: '#/definitions/value'     
    """
    text = request.args.get("IT_Section_Text")
    temp = [] #
    all_stopwords = stopwords.words('english')

    text_sent = nltk.sent_tokenize(text)

    for sentence in text_sent:
     new_sentence = re.sub('[^a-zA-Z]', ' ', sentence)
     new_sentence = new_sentence.lower()
     new_sentence = new_sentence.split()
     new_sentence= [wnlem.lemmatize(word) for word in new_sentence if not word in set(all_stopwords)]
     print(sentence)
     new_sentence = ' '.join(new_sentence)
     print(new_sentence)
     new_corpus = [new_sentence]
     new_X_test = cv.transform(new_corpus).toarray()
     new_y_pred = classifier.predict(new_X_test)
     print(new_y_pred)
     if new_y_pred == 0:
          x = 'IT_Act_2000_Section_66'
     else:
          x = 'IT_Act_2000_Section_67'

    temp.append([sentence,x])
    
         
    return str(temp)


@app.route('/predict_file', methods=["POST"])
def Analyse_Section_IT_ACT_File():
    """Example file endpoint returning a prediction of IT ACT SECTION
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    definitions:
        value:
            type: object
            properties:
                value_name:
                    type: string
                    items:
                        $ref: '#/definitions/Color'
        Color:
            type: string
    responses:
        200:
            description: OK
            schema:
                $ref: '#/definitions/value'   
    """
    file = request.files['file']

        
    
    """**Taking input data from the text file **

=> Text file name = SOME_INPUT_TEXT.txt 

=> UTF-8 ( Unicode Transformation Format ) one of the most commonly used encodings, and 8 bits values are used for encoding.
"""
    IT_SECTION = file.read()
    text = str(IT_SECTION)
    temp = [] #
    all_stopwords = stopwords.words('english')

    text_sent = nltk.sent_tokenize(text)

    for sentence in text_sent:
     new_sentence = re.sub('[^a-zA-Z]', ' ', sentence)
     new_sentence = new_sentence.lower()
     new_sentence = new_sentence.split()
     new_sentence= [wnlem.lemmatize(word) for word in new_sentence if not word in set(all_stopwords)]
     print(sentence)
     new_sentence = ' '.join(new_sentence)
     print(new_sentence)
     new_corpus = [new_sentence]
     new_X_test = cv.transform(new_corpus).toarray()
     new_y_pred = classifier.predict(new_X_test)
     print(new_y_pred)
     if new_y_pred == 0:
          x = 'IT_Act_2000_Section_66'
     else:
          x = 'IT_Act_2000_Section_67'
          
     
    temp.append([sentence,x])
         
    return str(temp)

 


if __name__ == '__main__':
    app.run()