import pandas as pd
import numpy as np
from ast import literal_eval #parse the stringified features into their corresponding python objects
    
def top_n_recommender(n, df, genre, percentile=0.95):
    if genre != 'All':
        s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'genres'
        gen_df = df.drop('genres', axis=1).join(s)
        df = gen_df[gen_df['genres'] == genre]
    
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & \
                    (df['vote_average'].notnull())]
    
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(n)
    
    return qualified

def nlp_based_recommenderer(n, df, title, cosine_sim):
    df = df.reset_index()
    titles = df['title']
    indices = pd.Series(df.index, index=df['title'])
    
    idx = str(indices[title])
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    
    return titles.iloc[movie_indices]  