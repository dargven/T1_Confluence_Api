import requests
import json
from config import BASE_URL, AUTH


class ConfluenceAPI:
    def __init__(self):
        self.base_url = BASE_URL
        self.auth = AUTH

    def get_pages(self, space_key, limit=100):
        url = f"{self.base_url}/content?spaceKey={space_key}&expand=body.storage&limit={limit}"
        response = requests.get(url, auth=self.auth)
        data = response.json()

        pages = []
        for page in data["results"]:
            pages.append({
                "id": page["id"],
                "title": page["title"],
                "content": page["body"]["storage"]["value"]
            })

        return pages

    def add_label_to_page(self, page_id, label):
        url = f"{self.base_url}/content/{page_id}/label"
        data = [{"prefix": "global", "name": label}]
        response = requests.post(url, json=data, auth=self.auth)
        return response.status_code

# Использование:
# api = ConfluenceAPI()
# documents = api.get_pages("DEV_DOCS")
