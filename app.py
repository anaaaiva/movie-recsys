import pandas as pd
import numpy as np
from ast import literal_eval

from validators import Max
import recommendation as r
import streamlit as st
from urllib.error import URLError

@st.cache_data
def get_data_top():
    df = pd.read_csv('the_movies_dataset/movies_metadata.csv')
    df['genres'] = df['genres'].fillna('[]').apply(literal_eval) \
                            .apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    
    return df[['title', 'year', 'vote_count', 'vote_average','genres']]

@st.cache_data
def get_data_desc():
    df = pd.read_csv("the_movies_dataset/preprocessed_movies_metadata_small.csv")
    cos_sim = pd.read_csv("the_movies_dataset/cos_sim.csv")
    return df, cos_sim

try:
    n = st.number_input('Insert a number of movies you would like to be recommended', min_value=1, max_value=100)
    st.write('The current number is ', n)
    
    df = get_data_top()
    genres = ['Animation','Comedy','Family','Adventure','Fantasy','Romance','Drama','Action', \
                'Crime','Thriller','Horror','History','Science Fiction','Mystery','War', 'Foreign', \
                'Music','Documentary','Western', 'All']
    genre = st.selectbox(
        "Choose genre:", genres)
    if not genre:
        st.error("Please select one genre.")
    else:    
        if genre == 'All':
            st.header(f'Top {n} most popular movies')
        else:
            st.header(f'Top {n} most popular {genre} movies') 
        data = st.dataframe(r.top_n_recommender(n, df, genre).drop('genres', axis=1), hide_index=True)
        
    df2, cos_sim = get_data_desc()
    films = df2['title']
    film = st.selectbox(
        "Choose title:", films)
    if not film:
        st.error("Please select one title.")
    else:
        st.header(f'Top {n} most similar movies to {film}')
        data2 = st.dataframe(r.nlp_based_recommenderer(n, df2, str(film), cos_sim), hide_index=True)
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )

