import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from main import app

client = app.test_client()

print("Testing /api/suggestions with OPTIONS preflight...")
res_opt = client.options('/api/suggestions', headers={'Origin': 'https://movierecommender-navy.vercel.app'})
print("OPTIONS Status:", res_opt.status_code)
print("Access-Control-Allow-Origin:", res_opt.headers.get('Access-Control-Allow-Origin'))

print("\nTesting /api/suggestions GET...")
res = client.get('/api/suggestions', headers={'Origin': 'https://movierecommender-navy.vercel.app'})
print("GET Status:", res.status_code)
print("Access-Control-Allow-Origin:", res.headers.get('Access-Control-Allow-Origin'))
print("Sample suggestions count:", len(res.json))

print("\nTesting /api/trending GET...")
res_trend = client.get('/api/trending', headers={'Origin': 'https://movierecommender-navy.vercel.app'})
print("Trending Status:", res_trend.status_code)
print("Access-Control-Allow-Origin:", res_trend.headers.get('Access-Control-Allow-Origin'))
