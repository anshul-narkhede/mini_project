import requests
import json
import time

API_URL = "http://127.0.0.1:8000/search"

# Sample search request payload
payload = {
    "query": "A quiet boutique hotel with a pool",
    "country": "India",
    "city": "Goa",
    "min_rating": 4.0,
    "top_k": 3
}

def test_search_api():
    print(f"Testing API endpoint: {API_URL}")
    print("Sending payload:", json.dumps(payload, indent=2))
    
    start_time = time.time()
    try:
        response = requests.post(API_URL, json=payload)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Success! Received 200 OK in {elapsed_time:.2f} seconds.")
            
            # Parse the response
            data = response.json()
            
            # Save the response to a JSON file
            output_file = "test_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
            print(f"✅ Test results successfully saved to {output_file}")
            print(f"Found {data.get('result_count')} matching hotels.")
        else:
            print(f"❌ Failed with status code: {response.status_code}")
            print("Response:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Failed to connect to the API. Is the server running on http://127.0.0.1:8000?")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    test_search_api()
