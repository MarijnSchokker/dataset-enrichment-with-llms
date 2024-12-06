import json
from pathlib import Path
from house_bot.enrichment.enrichment_types import HOUSES_CACHE
from house_bot.enrichment.enrichment_types import House, HouseFeatures


import pandas as pd


def fetch_housing_data_from_disk(cache_file: Path) -> pd.DataFrame:
    return pd.read_parquet(cache_file)


def save_housing_data_to_disk(houses: pd.DataFrame, cache_file: Path):
    houses.to_parquet(cache_file)


def fetch_housing_data(
    cache_file: Path,
    ignore_cache: bool,
) -> pd.DataFrame:
    if ignore_cache or not cache_file.exists():
        raise ValueError("cache file must exist")
    else:
        houses = fetch_housing_data_from_disk(cache_file=cache_file)

    return houses


def serialize_house(house: pd.Series) -> House:
    serialized_house = House(**house.to_dict())
    return serialized_house


def fetch_house_from_disk(house_id: str, cache_file: Path = HOUSES_CACHE) -> House:
    houses = fetch_housing_data_from_disk(cache_file=cache_file)
    houses = houses.set_index("id")
    house_pandas: pd.Series = houses.loc[house_id]
    house_pandas["id"] = house_id
    house = serialize_house(house_pandas)
    return house


def serialize_housing_data(houses: pd.DataFrame) -> list[House]:
    serialized_houses = [serialize_house(house) for index, house in houses.iterrows()]
    return serialized_houses


def save_house_features(house_features: HouseFeatures, json_path: Path):
    house_features_dict = house_features.model_dump()
    house_features_json = json.dumps(house_features_dict)
    json_path.parent.mkdir(exist_ok=True, parents=True)
    json_path.write_text(house_features_json)


def load_house_features(json_path: Path) -> HouseFeatures:
    house_features_json = json_path.read_text()
    house_features_dict = json.loads(house_features_json)
    house_features = HouseFeatures(**house_features_dict)
    return house_features