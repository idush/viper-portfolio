import requests

class NewsScraper:
    def get_latest_hackernews(self):
        url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        try:
            r = requests.get(url)
            ids = r.json()[:5]
            stories = []
            for story_id in ids:
                item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                item_res = requests.get(item_url)
                stories.append(item_res.json())
            return stories
        except Exception as e:
            return [{"error": str(e)}]
