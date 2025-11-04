import requests
import os

class BreachDataChecker:
    def check_email(self, email):
        api_key = os.getenv("HIBP_API_KEY", "")
        url = f"https://haveibeenpwned.com/api/v3/breachedaccount/{email}"
        headers = {"User-Agent": "OSINT-Toolkit"}
        if api_key:
            headers["hibp-api-key"] = api_key
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.status_code}
        except Exception as e:
            return {"error": str(e)}
