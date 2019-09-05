import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = open('videogame.txt', 'r', errors = 'ignore')
file = f.read()
file.lower()

sent_token = nltk.sent_tokenize(file)
word_token = nltk.word_tokenize(file)

lemmer = nltk.stem.WordNetLemmatizer()

def LemToken(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

rem_punc = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemToken(nltk.word_tokenize(text.lower().translate(rem_punc)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey", "how u doin", "heya", "yo")
GREETING_RESPONSES = ["hi", "hey", "heylo", "hi there", "hello", "hi whats up dude", "Hi nice talking to you"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    bot_response=''
    sent_token.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_token)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        bot_response+= "I am sorry! I don't understand you"
        return bot_response
    else:
        bot_response += sent_token[idx]
        return bot_response

flag = True
print("Bot: My name is GameBot. Made by Rajmeet I am trained to answer your queries about gaming. \nif you want to exit, just type 'bye'")

while(flag==True):
    user_response = input(">")
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you'):
            flag=False
            print("BOT: You are welcome")
        if user_response == "what are video games":
            for i in sent_token[1:2]:
                print(i, end = "\n")
        else:
            if(greeting(user_response)!=None):
                print("BOT: "+greeting(user_response))
            else:
                print("BOT: ",end="")
                print(response(user_response))
                sent_token.remove(user_response)
    else:
        flag=False
        print("BOT: Bye! take care")