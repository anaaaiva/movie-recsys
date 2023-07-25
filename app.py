import pandas as pd
import numpy as np
from ast import literal_eval
import recommendation as r
import streamlit as st
from urllib.error import URLError

@st.cache_data
def get_data():
    df = pd.read_csv('the_movies_dataset/movies_metadata.csv')
    df['genres'] = df['genres'].fillna('[]').apply(literal_eval) \
                            .apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    
    return df[['title', 'year', 'vote_count', 'vote_average','genres']]

try:
    df = get_data()
    genres = ['Animation','Comedy','Family','Adventure','Fantasy','Romance','Drama','Action', \
                'Crime','Thriller','Horror','History','Science Fiction','Mystery','War', 'Foreign', \
                'Music','Documentary','Western', 'All']
    genre = st.selectbox(
        "Choose genre:", genres)
    if not genre:
        st.error("Please select one genre.")
    else:    
        if genre == 'All':
            st.header(f'Top 10 most popular movies')
        else:
            st.header(f'Top 10 most popular {genre} movies') 
        data = st.dataframe(r.top_n_recommender(10, df, genre).drop('genres', axis=1), hide_index=True)
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )

