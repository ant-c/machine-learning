"""
@author  Anthony Cruz
@file    a_cruz_hw6.py
@date    2019-04-26
@brief   This program performs the Naive Bayes Classifier and Logistic Regression algorithm for text data.
"""
import pandas as pd
import sklearn.datasets as skd
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------------------
# Main entry-point for this application.
#
# @return Exit-code for the process - 0 for success, else an error code.
# ---------------------------------------------------------------------------

print("\n------------------ Enron Email Dataset ------------------")
print("Naive Bayes Classification:")
categories = ['ham', 'spam']
# loading all files as training data
email_train = skd.load_files('/Users/ant_c/Desktop/Enron-Emails', categories=categories, encoding='ISO-8859-1')
print("Total size of dataset:", len(email_train.data))
print("Target classes:", email_train.target_names, "\n")

# since there is only one dataset (training set) split it 75/25 to create test data
X_train, X_test, y_train, y_test = train_test_split(email_train.data, email_train.target, test_size=0.25, random_state=1)

# turn the text into vectors of numerical values suitable for statistical analysis
email_counts = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB())])
# train the model using the training sets
email_counts.fit(X_train, y_train)
# predict the test cases
predicted = email_counts.predict(X_test)

# print metrics
print(metrics.classification_report(predicted, y_test, target_names=email_train.target_names)),
metrics.confusion_matrix(predicted, y_test)
# print how often the classifier was correct
print("Accuracy:", metrics.accuracy_score(predicted, y_test))

# -----------------------------------------------------------------------------
print("\n------------------ Enron Email Dataset ------------------")
print("Logistic Regression:")
print("Total size of dataset:", len(email_train.data))
print("Target classes:", email_train.target_names, "\n")

# turn the text into vectors of numerical values suitable for statistical analysis and train
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

logistic = LogisticRegression(solver='lbfgs', max_iter=4000)
logistic.fit(X_train_dtm, y_train)
# make class predictions for X_test_dtm
y_pred_class = logistic.predict(X_test_dtm)

# print metrics
print(metrics.classification_report(y_pred_class, y_test, target_names=["ham", "spam"])),
metrics.confusion_matrix(y_pred_class, y_test)
# print how often the classifier was correct
print("Accuracy:", metrics.accuracy_score(y_pred_class, y_test))

# -----------------------------------------------------------------------------
print("\n------------------ SMS Spam Collection ------------------")
print("Naive Bayes Classification:")
sms = pd.read_csv("/Users/ant_c/Desktop/bigdata/spam.csv", encoding="ISO-8859-1", usecols=[0, 1], skiprows=1,
                  names=["label", "message"])

# csv file so map it accordingly
sms.label = sms.label.map({"ham": 0, "spam": 1})
print("Total size of dataset:", len(sms))
print("Target classes: ['ham', 'spam']", "\n")

# since there is only one dataset (training set) split it 75/25 to create test data
X_train, X_test, y_train, y_test = train_test_split(sms.message, sms.label, test_size=0.25, random_state=1)

# turn the text into vectors of numerical values suitable for statistical analysis
text = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB())])
# train the model using the training sets
text.fit(X_train, y_train)
# predict the test cases
predicted = text.predict(X_test)

# print metrics
print(metrics.classification_report(predicted, y_test, target_names=["ham", "spam"])),
metrics.confusion_matrix(predicted, y_test)
# print how often the classifier was correct
print("Accuracy:", metrics.accuracy_score(predicted, y_test))


# -----------------------------------------------------------------------------
print("\n------------------ SMS Spam Collection ------------------")
print("Logistic Regression:")
sms = pd.read_csv("/Users/ant_c/Desktop/bigdata/spam.csv", encoding="ISO-8859-1", usecols=[0, 1], skiprows=1,
                  names=["label", "message"])

print("Total size of dataset:", len(sms))
print("Target classes: ['ham', 'spam']", "\n")

# csv file so map it accordingly
sms['label_num'] = sms['label'].map({'ham': 0, 'spam': 1})
X = sms['message']
y = sms['label_num']

# since there is only one dataset (training set) split it 75/25 to create test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# turn the text into vectors of numerical values suitable for statistical analysis and train
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

logistic = LogisticRegression(solver='lbfgs')
logistic.fit(X_train_dtm, y_train)
# make class predictions for X_test_dtm
y_pred_class = logistic.predict(X_test_dtm)

# print metrics
print(metrics.classification_report(y_pred_class, y_test, target_names=["ham", "spam"])),
metrics.confusion_matrix(y_pred_class, y_test)
# print how often the classifier was correct
print("Accuracy:", metrics.accuracy_score(y_pred_class, y_test))

# -----------------------------------------------------------------------------
