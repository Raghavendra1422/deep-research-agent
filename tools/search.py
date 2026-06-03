import os
import requests

def search_web(query: str) -> list:
    """Search web using Serper API, returns list of results"""

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json"
    }

    payload = {
        "q": query,
        "num": 5          # 5 results per sub-topic
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    results = []

    # Extract organic results
    for item in data.get("organic", []):
        results.append({
            "title":   item.get("title", ""),
            "link":    item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })

    return results