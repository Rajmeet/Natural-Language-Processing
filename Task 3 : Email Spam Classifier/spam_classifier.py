''' Load libraries '''
import pandas as pd                                             # For Data Exploration, 
import numpy as np                                              # To create arrays
import nltk                                                     # For Text Pre-processing 
import re                                                       # For Text Pre-processing
from nltk.tokenize import word_tokenize                         # Tokenize text into words
from nltk.stem import PorterStemmer                             # Reducing word to it's root
from sklearn.feature_extraction.text import CountVectorizer     # Create Bag of Words
from sklearn.model_selection import train_test_split            # Split data into groups (Testing and Training)
from sklearn.naive_bayes import MultinomialNB                   # Selecting the Multinomial Algorithm 
from sklearn.metrics import accuracy_score                      # Display Accuracy 

from nltk.corpus import stopwords
from string import punctuation
trashwords = stopwords.words('english')

# Use in case you get an error trying to import stopwords
# nltk.download('stopwords') 

'''
Import your data into the program and display it

Task: Load dataset and display dataset
Hint: Using pandas will make your life a lot easier
'''
df = pd.read_csv('emails_small.csv')
print(df)

'''
Check for any Null Values (empty rows) and drop duplicate rows

Task: Eliminate empty and duplicate rows
Hint: Use pandas!
'''
df.drop_duplicates(inplace = True)
print(df.isnull().sum())

'''
Now it's time to start cleaning. Let's remove any unnecessary pieces of text.

Hint: Display one piece of text to see what we should remove
Task: Iterate over rows and perform cleaning, then display your dataset again
'''
# print(df['text'][0])

for index,row in df.iterrows():
    new_text = re.sub('Subject: |re : |fw : |fwd : ', '', row['text'])
    new_text = new_text.lower().strip()
    df.loc[index,'text'] = new_text

# print(df)
    
'''
Create your final corpus of sentences. The corpus must be a list of all sentences
in its stemmed form and should not include punctuation characters or stopwords.

Task: Create a list of strings containing each stemmed and processed sentence.
Hint: Tokenize each sentence to handle words separately. Use word_tokenize to
tokenize and PortStemmer() to stem.
'''
corpus = []
stemmer = PorterStemmer()
for text in df['text']:
    tokenized_text = nltk.word_tokenize(text)
    stemmed_text = ''
    for word in tokenized_text:
        if word not in punctuation and word not in trashwords:
            stemmed_text += stemmer.stem(word) + ' '
    corpus.append(stemmed_text)
    
# print(corpus[0])

'''
Create a Bag of Words representation of your corpus (x) and a list of the
labels (y). Both must have the same length!

Task: Create a Bag of Words model and its respective list of labels
Hint: Use scikit's CountVectorizer()
'''
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
y = df.iloc[:,1].values

# print(x)
# print(y)
# print(cv.get_feature_names())

'''
Split your data into a training set and a testing set. We chose 20% for the
test size, but you can tweak this value and see how it affects the final result.
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

'''
Classify your data using Naive Bayes algorithm.

Task: Create a Naive Bayes classifier using only your training data (i.e. 
x_train and y_train).
Hint: Use scikit's MultinomialNB()
'''
classifier = MultinomialNB()
classifier.fit(x_train,y_train)

'''
Measure the accuracy of your model with the testing data.

Task: Use your classifier to make predictions for x_test and then determine 
its accuracy in respect to y_test.
Hint: Use scikit's accuracy_score()
'''
y_pred = classifier.predict(x_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

'''
OPTIONAL
Make it accept user's input and determine whether or not the text entered is
spam.
'''
user_text = input('Input the text: ')
prediction = classifier.predict(cv.transform([user_text]))[0]
if prediction == 1:
    print('Spam!')
else:
    print('Not spam!')
