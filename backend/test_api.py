import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "features": [90, 42, 43, 20.8, 82.0, 6.5, 202.0]  # Sample values
}

response = requests.post(url, json=data)
print(response.json())  # Expected output: {"crop": "Rice"}
