# CyberCrime-Classification-using-Ensemble-Learning
Ensem_SLDR which can predict the relevant sections of IT Act 2000 from the compliant text with the aid of Natural Language Processing, Machine Learning, and Ensemble Learning methods

<h3 align="left">About the Project</h3>

With the advancement of technology, cybercrimes are surging at an alarming rate as
miscreants pour into the world's modern reliance on the virtual platform. Cyber-crime cells
in India, have a repository of detailed information on cybercrime which includes complaints
and investigations. Hence, there is a need for automation systems empowered by artificial
intelligence technologies for the analysis of cybercrime and their classification of sections.
The focus of this project is to construct a model, Ensem_SLDR which can predict the
relevant sections of IT Act 2000 from the compliant text with the aid of Natural Language
Processing, Machine Learning, and Ensemble Learning methods. In the proposed
methodology, Bag of Words approach is applied for performing feature engineering where
these features are given as input to the hybrid model Ensem_SLDR. The proposed model
is implemented using Support Vector Machine (SVM), Logistic Regression, Decision Tree,
and Random Forest and gave better performance by having 96.55 % as testing precision,
which is higher than the past model implemented using a single learning algorithm.


![Flowchart](/images/Minorproject_stack.jpg)

<h3 align="left">Functionality</h3>

At present, while registering an FIR or Investigation, Investigation
officers correlate the torts with the various Act under IT act 2000 and choose appropriate
Act & Sections. This necessitates IO to have prior deep knowledge and a clear
understanding of the criminal Law definitions and Interpretations. As the torts are
predefined (definitions), it is possible to predict the relevant sections from the compliant
text or FIR description or Investigation reports using Artificial Intelligence like Machine
Learning, NLP methodologies.
The API enables to perform the following functions in a broad sense:
* Analyze text
* Analyze file
* Summarize text

<h3 align="left">Tech Stack</h3>

* **Python:** The code for the text analyzer has been written in python
* **Natural Language Processing:** By using Natural Language Processing, we will make the computer truly understand more than just the objective definitions of the words. It includes using Bag of Words model which is a way of extracting features from the text for use in modeling.
* **Machine Learning:** A classifier or classification algorithm has been used to identify whether a given piece of text is Sections 66 and 67. 
* **NLTK:** NLTK (Natural Language Toolkit) is a popular open-source package in Python. Rather than building all tools from scratch, NLTK provides all common NLP Tasks. 
* **Flask:** Flask is a lightweight WSGI web application framework. It is designed to make getting started quick and easy, with the ability to scale up to complex applications.
* **Flasgger:** Flasgger is a Flask extension to help the creation of Flask APIs with documentation and live playground powered by SwaggerUI.
* **Postman:**  Postman is a great tool when trying to dissect RESTful APIs made by others or test ones we have made our self. It offers a sleek user interface with which to make HTML requests, without the hassle of writing a bunch of code just to test an API's functionality.

