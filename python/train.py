import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

data=pd.read_csv("dataset_demo.csv")

X=data['question']
y=data['intent']

vec=TfidfVectorizer()
X_vec=vec.fit_transform(X)

model=LinearSVC()
model.fit(X_vec,y)

pickle.dump(model,open("model.pkl","wb"))
pickle.dump(vec,open("vectorizer.pkl","wb"))

print("Train xong")