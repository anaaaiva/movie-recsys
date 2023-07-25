import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv('the_movies_dataset/preprocessed_movies_metadata.csv')

links_small = pd.read_csv('the_movies_dataset/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

data = data[data['id'].isin(links_small)]

tfidfvec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
tfidf_movieid = tfidfvec.fit_transform(data['soup'])

#print(tfidf_movieid.shape, type(tfidf_movieid)) (46628, 1623025) было без использования links_small

cos_sim = linear_kernel(tfidf_movieid, tfidf_movieid)

pd.DataFrame(cos_sim).to_csv("the_movies_dataset/cos_sim.csv", index=False)

data.to_csv("the_movies_dataset/preprocessed_movies_metadata_small.csv", index=False)



