import matplotlib.pyplot as plt
from textblob import TextBlob
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import re
nltk.download('punkt') 
nltk.download('averaged_perceptron_tagger') 
nltk.download('wordnet') 

path = r"INSERT PATH HERE"
path = pd.read_csv(path)

path = path[['text']]
path.columns = ['tweets']

stopwords = ['the', 'and', 'is', 'it', 'to', 'of', 'for', 'in', 'that', 'this', 'a', 'an']

def get_wordnet_pos(token):

    tag = nltk.pos_tag([token])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ,
                'N': wordnet.NOUN,
                'V': wordnet.VERB,
                'R': wordnet.ADV}
    
    return tag_dict.get(tag, wordnet.NOUN)

def cleanTweets(tweet):
    tweet = tweet.lower() 
    tweet = tweet.strip() 
    tweet = re.sub("#ethereum", 'ethereum', tweet) 
    tweet = re.sub("#Ethereum", 'Ethereum', tweet) 
    tweet = re.sub('#[A-Za-z0-9]+', '', tweet) 
    tweet = re.sub('\\n', '', tweet) 
    tweet = re.sub('https:\/\/\S+', '', tweet) 
    tweet = re.sub('@\w+', '', tweet) 
    tweet = re.sub('[^a-zA-Z0-9\s\.,!?;-]', '', tweet) 
    tweet = ' '.join([word for word in tweet.split() if word not in stopwords])
    lemmatizer = WordNetLemmatizer() 
    tokens = word_tokenize(tweet) 
    lem_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    tweet = ' '.join(lem_tokens)
    return tweet

path['cleaned_tweets'] = path['tweets'].apply(cleanTweets)


def setSubjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

def setPolarity(tweet):
    return TextBlob(tweet).sentiment.polarity

path['subjectivity'] = path['cleaned_tweets'].apply(setSubjectivity)
path['polarity'] = path['cleaned_tweets'].apply(setPolarity)

def setSentiment(score):
    if score < 0:
        return "negative"
    elif score == 0:
        return "neutral"
    else:
        return "positive"
    
path['sentiment'] = path['polarity'].apply(setSentiment)