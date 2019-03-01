import pandas as pd
import numpy as np
import sys

def features(name):
    name = name.lower()
    return {
        'first-letter': name[0], # First letter
        'first2-letters': name[0:2], # First 2 letters
        'first3-letters': name[0:3], # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }
#print (features("John"))
# {'first2-letters': 'jo', 'last-letter': 'n', 'first-letter': 'j', 'last2-letters': 'hn', 'last3-letters': 'ohn', 'first3-letters': 'joh'}
list_ = []
muslim=pd.read_csv('data/muslimNames.csv')
hindu=pd.read_csv('data/hinduNames.csv')
list_.append(muslim)
list_.append(hindu)
#list_.append(sikh)

names = pd.concat(list_)

input_name=raw_input('Enter name: ')
names = names.values
TRAIN_SPLIT = 0.8

# Vectorize the features function
features = np.vectorize(features)

# Extract the features for the whole dataset
X = features(names[:, 0]) # X contains the features

# Get the gender column
y = names[:, 2]           # y contains the targets

from sklearn.utils import shuffle
X, y = shuffle(X, y)
X_train, X_test = X[:int(TRAIN_SPLIT * len(X))], X[int(TRAIN_SPLIT * len(X)):]
y_train, y_test = y[:int(TRAIN_SPLIT * len(y))], y[int(TRAIN_SPLIT * len(y)):]

from sklearn.feature_extraction import DictVectorizer

#print (features(["Mary", "John"]))
vectorizer = DictVectorizer()
vectorizer.fit(X_train)

#transformed = vectorizer.transform(features(["Mary", "John"]))
#print (transformed)
#print (type(transformed)) # <class 'scipy.sparse.csr.csr_matrix'>
#print (transformed.toarray()[0][12])    # 1.0
#print (vectorizer.feature_names_[12])   # first-letter=m

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(vectorizer.transform(X_train), y_train)

import csv,codecs,io

input_name = input_name.split()[0]
		
result=(clf.predict(vectorizer.transform(features([input_name]))))
if "muslim"  in result:
	result='muslim'
else:
	result='hindu'
#finalres=input_name+','+result
print('Religion= '+result)