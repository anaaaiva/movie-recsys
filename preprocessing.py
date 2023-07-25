import pandas as pd
import numpy as np
from ast import literal_eval #parse the stringified features into their corresponding python objects
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import warnings; warnings.simplefilter('ignore')

movies_metadata = pd.read_csv('the_movies_dataset\movies_metadata.csv')
credits = pd.read_csv('the_movies_dataset/credits.csv')
keywords = pd.read_csv('the_movies_dataset/keywords.csv')

df = movies_metadata[['id', 'title', 'production_companies', 'genres', 'overview', 'tagline']]

df = df.drop([19730, 29503, 35587])
df.id = df.id.astype(int)
credits.id = credits.id.astype(int)
keywords.id = keywords.id.astype('int')

df = df.merge(credits, on='id')
df = df.merge(keywords, on='id')

df['title'] = df['title'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))

df['overview'] = df['overview'].fillna('')

df['tagline'] = df['tagline'].fillna('')

df['genres'] = df['genres'].fillna('[]').apply(literal_eval) \
                           .apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []) \
                           .apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

df['production_companies']= df['production_companies'].fillna('[]') \
                                                      .apply(literal_eval) \
                                                      .apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []) \
                                                      .apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

df['crew'] = df['crew'].apply(literal_eval).apply(get_director) \
                       .astype('str').apply(lambda x: str.lower(x.replace(" ", "")))

df['keywords'] = df['keywords'].apply(literal_eval) \
                               .apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

df['cast'] = df['cast'].apply(literal_eval) \
                       .apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []) \
                       .apply(lambda x: x[:5] if len(x) >=5 else x) \
                       .apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

s = df.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s = s[s > 1]

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

df['keywords'] = df['keywords'].apply(filter_keywords) \
                               .apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
                               
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
  
lemmatizer = WordNetLemmatizer()
  
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
  
VERB_CODES = {'VB',  # Verb, base form
              'VBD',  # Verb, past tense
              'VBG',  # Verb, gerund or present participle
              'VBN',  # Verb, past participle
              'VBP',  # Verb, non-3rd person singular present
              'VBZ',}  # Verb, 3rd person singular present

def preprocess_sentences(text):
    text = text.lower()
    temp_sent =[]
    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)
    for i, word in enumerate(words):
        if tags[i][1] in VERB_CODES: 
            lemmatized = lemmatizer.lemmatize(word, 'v')
        else:
            lemmatized = lemmatizer.lemmatize(word)
        if lemmatized not in stop_words and lemmatized.isalpha():
            temp_sent.append(lemmatized)
          
    finalsent = ' '.join(temp_sent)
    finalsent = finalsent.replace("n't", " not")
    finalsent = finalsent.replace("'m", " am")
    finalsent = finalsent.replace("'s", " is")
    finalsent = finalsent.replace("'re", " are")
    finalsent = finalsent.replace("'ll", " will")
    finalsent = finalsent.replace("'ve", " have")
    finalsent = finalsent.replace("'d", " would")
    return finalsent

df['preprocessed_overview'] = df['overview'].apply(preprocess_sentences)
df['preprocessed_tagline'] = df['tagline'].apply(preprocess_sentences)

idxs = ['keywords', 'cast', 'genres', 'production_companies', 'preprocessed_tagline', 'preprocessed_overview', 'crew', 'title']
for idx in idxs:
    df[idx] = df[idx].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x))

def concat_columns(row):
    return ' '.join(row)

df['soup'] = df[idxs].apply(concat_columns, axis=1)

df.to_csv('the_movies_dataset/preprocessed_movies_metadata.csv', index=False)