import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from main import app

client = app.test_client()

print("Fetching /api/movie/Avatar...")
res = client.get('/api/movie/Avatar')
print("Status:", res.status_code)
data = res.json

print(f"Movie: {data.get('title')}")
print(f"Director: {data.get('director_name')}")
print(f"Casts count: {len(data.get('casts', []))}")
print(f"First Cast: {data.get('casts', [])[0] if data.get('casts') else None}")
print(f"Recommended Movies count: {len(data.get('recommended_movies', []))}")
print(f"First Rec Movie: {data.get('recommended_movies', [])[0] if data.get('recommended_movies') else None}")

assert len(data.get('casts', [])) > 0, "Casts should not be empty!"
assert len(data.get('recommended_movies', [])) > 0, "Recommended movies should not be empty!"
print("\n[SUCCESS] Both Casts and Recommended Movies are populated properly as arrays!")
