import os
import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = r"c:\1_Paras Admissions\Projects\Movie_Recommender"
csv_path = os.path.join(BASE_DIR, 'csv', 'main_data1.csv')

print("Loading CSV...")
df = pd.read_csv(csv_path, encoding='latin1')
print(f"Loaded {len(df)} rows.")

print("Vectorizing...")
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['comb'])
print(f"count_matrix shape: {count_matrix.shape}, data memory size: {count_matrix.data.nbytes / (1024*1024):.2f} MB")

# Test dynamic on-demand similarity for a movie
test_movie = "the avengers"
t0 = time.time()
idx = df.loc[df['movie_title'] == test_movie].index[0]
sim_scores = cosine_similarity(count_matrix[idx], count_matrix).flatten()
top_indices = sim_scores.argsort()[::-1][1:11]
recommended = [df['movie_title'].iloc[i] for i in top_indices]
t1 = time.time()

print(f"Recommendation computation took: {(t1 - t0)*1000:.2f} ms")
print(f"Top 10 recommended for '{test_movie}': {recommended}")
