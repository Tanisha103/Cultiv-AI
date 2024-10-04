import requests
import pandas as pd
url = "https://api.kaggle datasets download -d snikhilrao/crop-disease-detection-dataset.com/data"  # Replace with the actual API endpoint
response = requests.get(url)
data = response.json()  # Assuming the response is in JSON format
df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)
