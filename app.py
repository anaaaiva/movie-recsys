import pandas as pd
import numpy as np
from ast import literal_eval

from validators import Max
import recommendation as r
import streamlit as st
import base64
from urllib.error import URLError

genres = ['Animation','Comedy','Family','Adventure','Fantasy','Romance','Drama','Action', \
                'Crime','Thriller','Horror','History','Science Fiction','Mystery','War', 'Foreign', \
                'Music','Documentary','Western', 'All']

#set image as a background
page_bg_img = '''
<style>
body {
background-image: url("https://wallpapercave.com/wp/wp1945897.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache_data
def get_data():
    df_top = pd.read_csv('the_movies_dataset/movies_metadata.csv')
    df_top['genres'] = df_top['genres'].fillna('[]').apply(literal_eval) \
                            .apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df_top['year'] = pd.to_datetime(df_top['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    
    df = pd.read_csv("the_movies_dataset/preprocessed_movies_metadata_small.csv")
    cos_sim = pd.read_csv("the_movies_dataset/cos_sim.csv")
    
    return df_top[['title', 'year', 'vote_count', 'vote_average','genres','overview']], df, cos_sim

try:
    n = st.number_input('Insert a number of movies', min_value=1, max_value=100, value=10)
    df_top, df, cos_sim = get_data()
    genre = st.selectbox(
        "Choose genre:", genres)
    if not genre:
        st.error("Please select one genre.")
    else:    
        if genre == 'All':
            st.header(f'Top {n} most popular movies')
        else:
            st.header(f'Top {n} most popular {genre} movies') 
        data = st.table(r.top_n_recommender(n, df_top, genre).drop(['genres', 'wr', 'vote_count', 'vote_average'], axis=1))
        
    films = df['title']
    film = st.selectbox(
        "Choose title:", films)
    if not film:
        st.error("Please select one title.")
    else:
        st.header(f'Top {n} most similar movies to {film}')
        #data2 = df.merge(df_top, on='title')
        data2 = st.table(pd.DataFrame(r.nlp_based_recommenderer(n, df, str(film), cos_sim)).merge(df_top, on='title').drop(['genres', 'vote_count', 'vote_average'], axis=1))
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )

