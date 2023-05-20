# Import required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
movies = [
    'The Shawshank Redemption',
    'The Godfather',
    'The Dark Knight',
    'Pulp Fiction',
    'Fight Club',
    'Forrest Gump',
    'Inception',
    'Goodfellas',
    'The Matrix',
    'Interstellar'
]

# Preprocess the movie titles
vectorizer = CountVectorizer().fit_transform(movies)
titles = movies

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(vectorizer)


def recommend_movies(movie_title):
    # Get the index of the movie title
    idx = titles.index(movie_title)

    # Get the similarity scores of the movie
    scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Recommend top 5 similar movies
    top_movies = [titles[idx] for idx, _ in sorted_scores[1:6]]
    return top_movies


# Test the recommendation system
movie_title = 'The Matrix'
recommended_movies = recommend_movies(movie_title)

print(f"Recommended movies for '{movie_title}':")
for movie in recommended_movies:
    print(movie)
