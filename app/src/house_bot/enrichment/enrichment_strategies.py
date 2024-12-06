
from house_bot.enrichment.enrichment_types import (
    House,
    HouseFeature,
    HouseFeatures,
    LLMMethod,
)
from house_bot.enrichment.llms import google_palm
from house_bot.enrichment.llms import openai_gpt


def enrich_house_with_mock(house: House) -> HouseFeatures:
    house_features = HouseFeatures(
        is_pet_friendly=HouseFeature(was_extracted=False),
        has_garden=HouseFeature(was_extracted=False),
        has_parking_place=HouseFeature(was_extracted=False),
        is_suitable_for_couple=HouseFeature(was_extracted=False),
        has_balcony=HouseFeature(was_extracted=False),
    )
    return house_features


def enrich_house_with_llm(
    house: House, method: LLMMethod = LLMMethod.gpt_35
) -> HouseFeatures:
    match method:
        case LLMMethod.mock:
            return enrich_house_with_mock(house)
        case LLMMethod.gpt_35_json_example:
            return openai_gpt.json_example(
                house,
                model="gpt-3.5-turbo-0613",
                deployment_id="gpt-35-turbo-us",
            )
        case LLMMethod.gpt_35_pydantic_schema:
            return openai_gpt.pydantic_schema(
                house,
                model="gpt-3.5-turbo-0613",
                deployment_id="gpt-35-turbo-us",
            )
        case LLMMethod.gpt_35:
            return openai_gpt.function_calling(
                house,
                model="gpt-3.5-turbo-0613",
                deployment_id="gpt-35-turbo-us",
            )
        case LLMMethod.gpt_4_json_example:
            return openai_gpt.json_example(
                house,
                model="gpt-4",
                deployment_id="gpt-4-us",
            )
        case LLMMethod.gpt_4_pydantic_schema:
            return openai_gpt.pydantic_schema(
                house,
                model="gpt-4",
                deployment_id="gpt-4-us",
            )
        case LLMMethod.gpt_4:
            return openai_gpt.function_calling(
                house,
                model="gpt-4",
                deployment_id="gpt-4-us",
            )
        case LLMMethod.palm_2_json_example:
            return google_palm.json_schema(house)
        case LLMMethod.palm_2: # pydantic schema
            return google_palm.pydantic_schema(house)
        case _:
            raise ValueError(f"Unknown method {method}")
