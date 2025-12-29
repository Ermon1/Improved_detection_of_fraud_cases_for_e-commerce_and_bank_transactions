import requests
import pandas as pd
import json
import numpy as np

# Create dummy data with 195 features (matching the model input)
# We'll just use random data for testing connectivity
features = [0.0] * 195
# Make it a list of dicts as expected by our API?
# Our API expects "features": List[Dict[str, Any]] -> converted to DataFrame
# But wait, if we pass a list of dicts, keys become columns.
# We need to match the 195 columns.
# It's better if we pass a structure that can be easily converted to the right DataFrame.
# If the model was trained on 'col1', 'col2'... we need those keys.
# Since we don't know the exact column names effectively (one-hot encoded), 
# we might fail if we just pass random keys.
# The API converts input to DataFrame.
# If the model expects specific columns, we must provide them.

# Let's try to load the columns from the training data if available, or just send a dummy request 
# and see if it fails with a "feature mismatch" error, which confirms the API is reachable.

url = 'http://localhost:5000/predict'
headers = {'Content-Type': 'application/json'}

# Create a single record
data = {
    "features": [{"col_" + str(i): 0.0 for i in range(195)}] 
}
# This will likely fail model prediction but test the API endpoint logic.

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Connection failed: {e}")
