import streamlit as st
import numpy as np
import pandas as pd
import os
import tarfile
import urllib
import email
import email.policy
from collections import Counter
import re
import nltk
from nltk.stem import PorterStemmer
import urlextract
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score


# Function to get email structure
def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([get_email_structure(sub_email) for sub_email in payload]))
    else:
        return email.get_content_type()

# Function to count email structures
def structure_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

# Function to convert HTML to plain text
def html_to_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return email.policy.structures

# Function to convert email to text
def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if ctype != 'text/plain' and ctype != 'text/html':
            continue
        try:
            content = part.get_content()
        except:
            content = part.get_payload()
        if ctype == 'text/plain':
            return content
        else:
            html = content
    if html:
        return html_to_text(content)

# Custom Transformer: EmailToWordCountTransformer
class EmailToWordCountTransformer (BaseEstimator, TransformerMixin):
    
    def __init__(self, strip_header = True, lowercase = True, replace_number = True,
                replace_urls = True, strip_puncuation = True, stemming = True):
        self.strip_header = strip_header
        self.lowercase = lowercase
        self.replace_number = replace_number
        self.replace_urls = replace_urls
        self.strip_puncuation =strip_puncuation
        self.stemming = stemming
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y = None):
        X_transformed = []
        if isinstance(X, list):
            for email in X:
                text = email_to_text(email) or ''
        else:
            text = email_to_text(X) or ''
            
            if self.lowercase:
                text = text.lower()
                
            if self.replace_urls:
                url_extractor = urlextract.URLExtract()
                urls = list(set(url_extractor.find_urls(text)))
                for url in urls:
                    text = text.replace(url, "URLs")
                    
            if self.replace_number:
                text = re.sub(r'\d+', 'NUMBER', text, flags = re.M)
                
            if self.strip_puncuation:
                text = re.sub(r'\W+', ' ', text, flags = re.M)
                
            word_counts = Counter(text.split())
            if self.stemming:
                stemming_word_counts = Counter()
                stemmer = PorterStemmer()
                for word, count in word_counts.items():
                    stemming_word_counts[stemmer.stem(word)] += count
                word_counts = stemming_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)
        

class WordCountToVectorTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, vocabulary_limit = 1000):
        self.vocabulary_limit = vocabulary_limit
        
    def fit(self, X, y = None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_limit]
        self.vocabulary_ = {word : index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y = None):
        rows = []
        col = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                col.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, col)), shape = (len(X), self.vocabulary_limit + 1))

import streamlit as st
import pickle
import warnings
from email.message import EmailMessage

warnings.filterwarnings("ignore", category=DeprecationWarning)
st.set_page_config(page_title='Spam Classification')
st.title('Spam Classification Model')

# Define custom CSS styles
CUSTOM_CSS = """
body {
    background-color: #f5f5f5;
    font-family: Arial, sans-serif;
    line-height: 1.5;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #ffffff;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

.title {
    text-align: center;
    font-size: 32px;
    color: #4c8bf5;
    margin-bottom: 30px;
}

.text-area {
    height: 400px;
    width: 100%;
    padding: 10px;
    font-size: 16px;
    border: 2px solid #ccc;
    border-radius: 10px;
}

.button {
    background-color: #4c8bf5;
    color: white;
    padding: 10px 20px;
    font-size: 18px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}

.result {
    margin-top: 20px;
    font-size: 20px;
    text-align: center;
    padding: 10px;
    border-radius: 5px;
}

.ham-result {
    background-color: #e5f1ea;
    color: #4a8d55;
}

.spam-result {
    background-color: #f8dede;
    color: #b94a48;
}
"""

st.markdown(f'<style>{CUSTOM_CSS}</style>', unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def get_model():
    model = pickle.load(open('email_spam.pickle', 'rb'))
    return model

email_message = EmailMessage()
email_message["From"] = "sender@example.com"
email_message["To"] = "recipient@example.com"
email_message["Subject"] = "Hello, Streamlit!"
email_message.set_content("This is a test email.")


email_string = email_message.as_string()



user_input = st.text_area("Enter the email",email_message, max_chars=1000, key="large_textbox", height=400, help="Type or paste the email content here")

input_email_message = EmailMessage()
input_email_message.set_content(user_input)


if st.button("Check Spam"):
    model = get_model()
    predictions=get_model().predict(input_email_message)
    predictions=int(predictions)
    
    if predictions == 0:
        st.markdown('<div class="result ham-result">Email is HAM</div>', unsafe_allow_html=True)
    elif predictions == 1:
        st.markdown('<div class="result spam-result">Email is SPAM</div>', unsafe_allow_html=True)

st.sidebar.title("About")
st.sidebar.info(
    "This is a web application that uses a machine learning model to classify "
    "whether an email is spam or not. It is built using Streamlit."
)

st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    1. Enter the email content in the text area on the left.
    2. Click the "Check Spam" button to classify the email.
    3. The result (HAM or SPAM) will be displayed below the button.
    """
)

st.sidebar.title("Additional Information")
st.sidebar.markdown(
    "The model used in this application is trained on a dataset of emails "
    "and is capable of predicting whether an email is spam or not."
)
