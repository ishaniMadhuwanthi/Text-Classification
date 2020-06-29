"""
E/15/211
CO544-Machine Learning and Data Mining
Text classification
"""

# import libraries
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
nltk.download('wordnet')

# imporitng the dataset
movie_data = load_files(r"txt_sentoken")
X, y = movie_data.data, movie_data.target

#text preprocessing
documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):

    document = re.sub(r'\W', ' ', str(X[sen]))    # Remove all the special characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)  # remove all single characters
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)  # Remove single characters from the start
    document = re.sub(r'\s+', ' ', document, flags=re.I)  # Substituting multiple spaces with single space
    document = re.sub(r'^b\s+', '', document)  # Removing prefixed 'b'
    document = document.lower()  # Converting to Lowercase

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)


# converting text to numbers(BoW)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

# Finding TFIDF
# Term frequency = (Number of Occurrences of a word)/(Total words in the document)
# IDF(word) = Log((Total number of documents)/(Number of documents containing the word))

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

# split into train and test data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training Text Classification Model
# ..............Random forest model.....................
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

# predict the sentiment for the documents in the test set
y_pred = classifier.predict(X_test)

# Evaluating the Model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

"""
output:-
[[180  28]
 [ 30 162]]
              precision    recall  f1-score   support

           0       0.86      0.87      0.86       208
           1       0.85      0.84      0.85       192

    accuracy                           0.85       400
   macro avg       0.85      0.85      0.85       400
weighted avg       0.85      0.85      0.85       400

0.855
"""

# ............support vector machine model..................
from sklearn import svm #import svm model

clf = svm.SVC(kernel='linear') # create a svm classifier
clf.fit(X_train, y_train) # train the model using the train data set
y_pred = clf.predict(X_test) #predict the response for test data set
print(accuracy_score(y_test,y_pred))

# ...............Naïve Bayesian Classifier...............
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Now we can use the model to predict classifications for our test features.
y_pred = nb.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# ...........Logistic Regression Model.............

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)

lr = grid.best_estimator_
lr.fit(X_train, y_train)
lr.predict(X_test)
print("Score: {:.2f}".format(lr.score(X_test, y_test)))
