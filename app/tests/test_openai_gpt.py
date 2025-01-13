"""A placeholder module for tests"""
import pytest
from src.house_bot.enrichment.llms.openai_gpt import json_example
from src.house_bot.enrichment.enrichment_types import House

def test_openai_gpt_json_example():
    """A simple test used for testing the openai_gpt.json_example function"""
    house = House(
        id="aaaaaaaa",
        description="A lovely house with a garden and parking place. It is suitable for a couple and has a balcony.",
        url="https://www.example.com",
        price=1000,
        surface_area=100,
        rent_or_buy="rent",
        city="Amsterdam",
        neighborhood="Oud-West",
        interior="furnished",
        rooms=3,
        publish_time="2022-01-01",
        zip_code="1234AB",
        street="Example Street",
        house_type="apartment",
    )
    model = "gpt-4o-mini"
    house_features = json_example(house, model)
    assert house_features.is_pet_friendly.was_extracted == False
    assert house_features.has_garden.was_extracted == False
    assert house_features.has_parking_place.was_extracted == False
    assert house_features.is_suitable_for_couple.was_extracted == True
    assert house_features.has_balcony.was_extracted == True

def test_pytest():
    """A simple test used for testing the pytest pipeline set-up"""
    assert True
