from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import urllib.parse
from firebase_admin import credentials

from instructor import OpenAISchema
import pandas as pd
from pydantic import Field
from typing import Optional


class LLMMethod(str, Enum):
    """
    Enum representing different methods for interacting with language models.

    Attributes:
        mock (str): A mock enrichment function.
        LLMMethod.gpt_4omini_json_example: Uses a JSON schema template in the prompt for the gpt-4o-mini model
        LLMMethod.gpt_4omini_pydantic_schema: Uses a Pydantic schema template in the prompt for the gpt-4o-mini model
        LLMMethod.gpt_4omini: Performs function calling with the gpt-4o-mini model.
        LLMMethod.gpt_4_json_example: Uses a JSON schema template in the prompt for the gpt-4 model.
        LLMMethod.gpt_4_pydantic_schema: Uses a Pydantic schema template in the prompt for the gpt-4 model.
        LLMMethod.gpt_4: Performs function calling with the gpt-4 model.

    Notes:
        - The previously used gpt-3.5-turbo-0613 model has been replaced by gpt-4o-mini, based on provider's recommendations.
    """
    mock = "mock"
    
    # gpt_35_json_example = "gpt-35-json-example"
    # gpt_35_pydantic_schema = "gpt-35-pydantic-schema"
    # gpt_35 = "gpt-35"
    # replaced above
    gpt_4omini_json_example = "gpt-4omini-json-example"
    gpt_4omini_pydantic_schema = "gpt-4omini-pydantic-schema"
    gpt_4omini = "gpt-4omini" # function calling
    
    gpt_4_json_example = "gpt-4-json-example"
    gpt_4_pydantic_schema = "gpt-4-pydantic-schema"
    gpt_4 = "gpt-4" # function calling

    palm_2_json_example = "palm-2-json-example"
    palm_2 = "palm-2" # pydantic schema


class HouseFeature(OpenAISchema):
    was_extracted: bool = Field(description="Whether the feature was extracted")
    quote: Optional[str] = Field(
        default=None, description="Exact quote from the listing"
    )
    value: Optional[bool] = Field(
        default=None, description="Whether the house has the feature"
    )


class HouseFeatures(OpenAISchema):
    """Correctly extracted house listing features"""

    is_pet_friendly: HouseFeature
    has_garden: HouseFeature
    has_parking_place: HouseFeature
    is_suitable_for_couple: HouseFeature
    has_balcony: HouseFeature


@dataclass
class House:
    id: str
    description: str
    url: str
    price: float
    surface_area: float
    rent_or_buy: str
    city: str
    neighborhood: str
    interior: str
    rooms: int
    publish_time: pd.Timestamp
    zip_code: str
    street: str
    house_type: str

    # optional fields
    is_available: bool | None = None
    city_slug: str | None = None
    pubsub_message_id: str | None = None
    street_slug: str | None = None
    image_url: str | None = None
    real_estate_agent_url: str | None = None
    real_estate_agent: str | None = None
    relative_url: str | None = None
    price_annotation: str | None = None
    house_type_slug: str | None = None
    coordinates: str | None = None
    longitude: str | None = None
    latitude: str | None = None

    # extracted features
    is_pet_friendly: HouseFeature | None = None
    has_garden: HouseFeature | None = None
    has_parking_place: HouseFeature | None = None
    is_suitable_for_couple: HouseFeature | None = None
    has_balcony: HouseFeature | None = None

    @property
    def google_maps_query(self):
        maps_query: str = urllib.parse.urlencode(
            {
                "q": f"{self.street}, {self.neighborhood}",
            }
        )
        return maps_query

    def __repr__(self):
        str_repr = f"🏡 {self.house_type} aan de {self.street} (€{self.price})\n\n"
        str_repr += "## 📝 Description\n"
        str_repr += f"{self.description.strip()}\n\n"
        str_repr += "## Eigenschappen\n" # TODO: this should be english
        str_repr += f"🏙 Neighborhood: {self.neighborhood}\n"
        str_repr += f"🛏 Rooms: {self.rooms}\n"
        str_repr += f"⬜️ Surface area: {self.surface_area} m²\n"
        str_repr += f"🏢 Real estate agent: {self.real_estate_agent}\n"
        str_repr += f"🌐 URL: {self.url}\n"
        str_repr += (
            f"📍 Google Maps link: https://maps.google.com/?{self.google_maps_query}\n"
        )

        # House features
        house_features_strs = []
        if self.is_pet_friendly is not None and self.is_pet_friendly.value is True:
            house_features_strs.append("🐕 pet friendly")
        if self.has_garden is not None and self.has_garden.value is True:
            house_features_strs.append("🌳 has garden")
        if self.has_parking_place is not None and self.has_parking_place.value is True:
            house_features_strs.append("🚗 has parking place")
        if (
            self.is_suitable_for_couple is not None
            and self.is_suitable_for_couple.value is True
        ):
            house_features_strs.append("👫 suitable for couple")
        if self.has_balcony is not None and self.has_balcony.value is True:
            house_features_strs.append("🌆 has balcony")
        house_features_str = ", ".join(house_features_strs)
        str_repr += f"{house_features_str}"

        return str_repr

# TODO: differences with scraper:
# FIRESTORE_CREDENTIAL used to be here

CACHE_FOLDER_NAME = os.environ.get("CACHE_FOLDER_NAME", "./data")
HOUSES_CACHE = Path(CACHE_FOLDER_NAME) / "houses.parquet"