import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from main import app

client = app.test_client()

print("Testing /api/movie/Avatar for streaming platforms...")
res = client.get('/api/movie/Avatar')
data = res.json
print("Streaming availability count:", len(data.get('streaming_availability', [])))
if data.get('streaming_availability'):
    first = data.get('streaming_availability')[0]
    print("First provider payload:", first)
    assert isinstance(first, dict), "Provider should be a dictionary!"
    assert 'provider_name' in first, "provider_name key should exist!"
    assert 'logo_path' in first, "logo_path key should exist!"
print("[SUCCESS] Streaming availability format verified!")
