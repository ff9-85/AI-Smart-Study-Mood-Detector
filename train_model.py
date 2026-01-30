import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

#Load dataset
data=pd.read_csv("sample dataset.csv")

#preprocess text
def preprocess(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]','',text)
    return text

data['text']=data['text'].apply(preprocess)
X=data['text']
y=data['mood']

#Vectorize
vectorizer=TfidfVectorizer()
X_vectorized=vectorizer.fit_transform(X)

#Train model
model=LogisticRegression()
model.fit(X_vectorized,y)

#Save model
pickle.dump(model,open("mood_model.pkl","wb"))
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))

print("✅ Model trained and saved!")

