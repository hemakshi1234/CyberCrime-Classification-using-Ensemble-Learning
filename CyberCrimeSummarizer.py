# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:29:36 2020

@author: Hemakshi Pandey
"""
from flask import Flask, request
from flasgger import Swagger
import nltk
# NLTK is a leading platform for building Python programs to work with human language data.
import re
#This module provides regular expression matching operations
from nltk.corpus import stopwords
nltk.download('stopwords')
# One of the major forms of pre-processing is to filter out useless data. 
#In natural language processing, useless words (data), are referred to as stop words.
nltk.download('punkt')
#This tokenizer divides a text into a list of sentences, by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences.
import heapq



app = Flask(__name__)
swagger = Swagger(app)

@app.route('/summarize_text', methods=["GET"])
def Summarize_Cyber_Crime():
    """Example endpoint returning a summarized text
    ---
    parameters:
      - name: no_of_lines
        in: query
        type: number
        required: true
      - name: Cyber_Crime_Text
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
    Number_of_Lines = request.args.get("no_of_lines")
    text = request.args.get("Cyber_Crime_Text")
    temp = [] #
   # Preprocessing the data
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text = re.sub(r'\s+',' ',text)

    clean_text = text.lower()
    clean_text = re.sub(r'\W',' ',clean_text)
    clean_text = re.sub(r'\d',' ',clean_text)
    clean_text = re.sub(r'\s+',' ',clean_text)

    print(clean_text)



    # Tokenize sentences
    sentences = nltk.sent_tokenize(text)


    # Stopword list
    stop_words = nltk.corpus.stopwords.words('english')

    # Word counts 
    word2count = {}
    for word in nltk.word_tokenize(clean_text):
      if word not in stop_words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1


     # Converting counts to weights
    max_count = max(word2count.values())
    for key in word2count.keys():
     word2count[key] = word2count[key]/max_count


     # Product sentence scores    
    sent2score = {}
    for sentence in sentences:
       for word in nltk.word_tokenize(sentence.lower()):
         if word in word2count.keys():
            if len(sentence.split(' ')) < 25:
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]


    # Gettings best 5 lines             
    best_sentences = heapq.nlargest(int(Number_of_Lines), sent2score, key=sent2score.get)

    print('---------------------------------------------------------')
    for sentence in best_sentences:
       temp.append(sentence)
      
             
    return str(temp)
  
    


@app.route('/summarize_file', methods=["POST"])
def Summarize_Cyber_Crime_File():
    """Example file endpoint returning a summarized text
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      - name: no_of_lines
        in: query
        type: number
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
    Number_of_Lines = request.args.get("no_of_lines")

        
    
    """**Taking input data from the text file **

=> Text file name = SOME_INPUT_TEXT.txt 

=> UTF-8 ( Unicode Transformation Format ) one of the most commonly used encodings, and 8 bits values are used for encoding.
"""
    IT_SECTION = file.read()
    text = str(IT_SECTION)
    temp = [] #
    # Preprocessing the data
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text = re.sub(r'\s+',' ',text)

    clean_text = text.lower()
    clean_text = re.sub(r'\W',' ',clean_text)
    clean_text = re.sub(r'\d',' ',clean_text)
    clean_text = re.sub(r'\s+',' ',clean_text)

    print(clean_text)



    # Tokenize sentences
    sentences = nltk.sent_tokenize(text)



    # Stopword list
    stop_words = nltk.corpus.stopwords.words('english')

    # Word counts 
    word2count = {}
    for word in nltk.word_tokenize(clean_text):
      if word not in stop_words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1


     # Converting counts to weights
    max_count = max(word2count.values())
    for key in word2count.keys():
     word2count[key] = word2count[key]/max_count



     # Product sentence scores    
    sent2score = {}
    for sentence in sentences:
       for word in nltk.word_tokenize(sentence.lower()):
         if word in word2count.keys():
            if len(sentence.split(' ')) < 25:
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]


    # Gettings best 5 lines             
    best_sentences = heapq.nlargest(int(Number_of_Lines), sent2score, key=sent2score.get)

    print('---------------------------------------------------------')
    for sentence in best_sentences:
       temp.append(sentence)
      
             
    return str(temp)
         
          


if __name__ == '__main__':
    app.run()





