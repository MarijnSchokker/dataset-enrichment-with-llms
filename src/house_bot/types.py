import pandas as pd

from playwright.async_api._generated import Page, Response as PlaywrightResponse


class HousingPlatform:
    @property
    def url_prefix(self) -> str:
        raise NotImplementedError

    def parse_houses(self, html: str) -> pd.DataFrame:
        raise NotImplementedError

    def parse_house_details(self, html: str) -> pd.Series:
        raise NotImplementedError

    async def is_on_captcha_page(self, response: PlaywrightResponse) -> bool:
        raise NotImplementedError

    async def solve_captcha(page: Page) -> bool:
        raise NotImplementedError
