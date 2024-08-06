import argparse
import requests

def predict_sentiment(text):
    url = "http://127.0.0.1:5001/predict"
    headers = {"Content-Type": "application/json"}
    data = {"text": text}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        print("Prediction result:")
        print(response.json())
    else:
        print(f"Error occurred while making the prediction request. Status code: {response.status_code}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict sentiment using a REST API.')
    parser.add_argument('text', type=str, help='Text to analyze')
    args = parser.parse_args()

    # Join all arguments together as the input text
    input_text = ' '.join(args.text)

    predict_sentiment(input_text)
