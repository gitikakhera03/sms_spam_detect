import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from scipy.sparse import csr_matrix
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
# tfidf = pickle.load(open(r'c:\Users\hp\Untitled Folder 1\sms-spam-classifier\vectorizer.pkl', 'rb'))
# model = pickle.load(open(r'c:\Users\hp\Untitled Folder 1\sms-spam-classifier\model.pkl', 'rb'))


model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB

# # Load the vectorizer and model
# model_path = 'c:\\Users\\hp\\Untitled Folder 1\\sms-spam-classifier\\model.pkl'
# with open(model_path, 'rb') as file:
#     vectorizer, model = pickle.load(file)

# # Example input
# input_message = ["Your input message here"]

# # Transforming the input message using the vectorizer
# vector_input = vectorizer.transform(input_message)

# # Making a prediction
# result = model.predict(vector_input)[0]
# print("Prediction:", result)
