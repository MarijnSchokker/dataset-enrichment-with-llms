from house_bot.fetch_from_api import fetch_html_from_api
from house_bot.platforms import get_platform
from house_bot.types import HousingPlatform


import pandas as pd


import os


def add_house_details(house: pd.Series) -> pd.Series | None:
    is_on_gcp: bool = os.environ.get("IS_ON_GCP", "false") == "true"
    api_url: str = os.environ.get("API_URL", "http://localhost:80")

    # scrape HTML page
    url: str = house["url"]
    html: str = fetch_html_from_api(url, api_url, gcp=is_on_gcp)

    # Initialize housing platform
    housing_platform: HousingPlatform = get_platform("pararius")

    # parse house details and update house
    try:
        house_details: pd.Series = housing_platform.parse_house_details(html)
        print(f"✅ added description to {house['id']}")
        return house_details
    except ConnectionError:
        house_details = pd.Series({"is_available": False})
        print("⚠️ house no longer available (setting is_available=False)")
        return house_details
    except ValueError:
        print("⚠️ failed to find description div")
        return None