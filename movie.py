import pandas as pd
import numpy as np
import os
import streamlit as st

movies=pd.read_csv("movies.csv")   
ratings=pd.read_csv("ratings.csv")

df=ratings.merge(movies,on="movieId",how="left")

movies["year"]=movies["title"].str.extract(r"\((\d{4})\)")
movies["year"]=pd.to_numeric(movies["year"], errors="coerce")

df=ratings.merge(movies, on="movieId", how="left")
movie_stats=(
    df.groupby("title")
    .agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
        genres=("genres", "first"),
        year=("year", "first")
    )
    .reset_index()
)
C=movie_stats["avg_rating"].mean()
m=movie_stats["rating_count"].quantile(0.70)

qualified = movie_stats[movie_stats["rating_count"] >= m].copy()

qualified["weighted_rating"]=(
    (qualified["rating_count"]/(qualified["rating_count"]+m)) * qualified["avg_rating"]
    + (m/(qualified["rating_count"] + m))*C
)

qualified["popularity_score"]=qualified["weighted_rating"]* np.log1p(qualified["rating_count"])

top_50_movies=(
    qualified.sort_values("popularity_score", ascending=False)
    .loc[:, ["title", "year", "genres", "avg_rating", "rating_count", "weighted_rating", "popularity_score"]]
    .head(50)
    .reset_index(drop=True)
)
print(top_50_movies.head(10))

os.makedirs("outputs", exist_ok=True)

top_50_movies.to_csv("outputs/top_50_popular_movies.csv", index=False)

genre_df=movie_stats.copy()
genre_df["genre_list"] = genre_df["genres"].str.split("|")
genre_df=genre_df.explode("genre_list")

genre_df=genre_df.merge(
    qualified[["title", "weighted_rating", "popularity_score"]],
    on="title",
    how="left"
)

def recommend_movies_by_genre(genre_name, top_n=5, min_votes=50):
    genre_name=genre_name.strip().lower()

    genre_movies=genre_df[
        genre_df["genre_list"].str.lower()==genre_name
    ].copy()

    genre_movies=genre_movies[genre_movies["rating_count"]>=min_votes]

    if genre_movies.empty:
        return pd.DataFrame(columns=[
            "title", "year", "genres", "avg_rating",
            "rating_count", "weighted_rating", "popularity_score"
        ])

    recs=(
        genre_movies.sort_values(
            by=["popularity_score", "weighted_rating", "rating_count"],
            ascending=False
        )
        .loc[:, ["title", "year", "genres", "avg_rating", "rating_count", "weighted_rating", "popularity_score"]]
        .drop_duplicates(subset=["title"])
        .head(top_n)
        .reset_index(drop=True)
    )
    return recs

st.title("Movie Recommender")
genre_choice=st.selectbox(
    "Choose a genre",
    sorted(genre_df["genre_list"].dropna().unique())
)

top_n=st.slider("Number of recommendations", 5, 10, 5)
min_votes = st.slider("Minimum votes", 10, 200, 50)

if st.button("Recommend"):
    recs = recommend_movies_by_genre(genre_choice, top_n=top_n, min_votes=min_votes)
    st.dataframe(recs)
