
from house_bot.enrichment.enrichment_types import (
    House,
    HouseFeature,
    HouseFeatures,
    LLMMethod,
)
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
    house: House, method: LLMMethod = LLMMethod.gpt_4omini
) -> HouseFeatures:
    """
    Enrich a given House object using a specified LLM model (LLMMethod).

    Parameters:
        house (House): The house object to be enriched. It should conform to the House class structure.
        method (LLMMethod, optional): The method to use for enrichment. 
                                       Defaults to LLMMethod.gpt_4omini. 
                                       Options include:
                                       - LLMMethod.mock: Uses a mock enrichment function.
                                       - LLMMethod.gpt_4omini_json_example: Uses a JSON schema template in the prompt for the gpt-4o-mini model
                                       - LLMMethod.gpt_4omini_pydantic_schema: Uses a Pydantic schema template in the prompt for the gpt-4o-mini model
                                       - LLMMethod.gpt_4omini: Performs function calling with the gpt-4o-mini model.
                                       - LLMMethod.gpt_4_json_example: Uses a JSON schema template in the prompt for the gpt-4 model.
                                       - LLMMethod.gpt_4_pydantic_schema: Uses a Pydantic schema template in the prompt for the gpt-4 model.
                                       - LLMMethod.gpt_4: Performs function calling with the gpt-4 model.

    Returns:
        HouseFeatures: The extracted house features returned by the selected model.

    Raises:
        ValueError: If an unknown method is specified.

    Example:
        enriched_house = enrich_house_with_llm(my_house, LLMMethod.gpt_4)
    Notes:
        - The previously used gpt-4 model has been replaced by gpt-4o-mini, gpt-4 is no longer available.
        - The previously used PaLM 2 model has been removed, as it will soon be deprecated.
    """
    match method:
        case LLMMethod.mock:
            return enrich_house_with_mock(house)
        case LLMMethod.gpt_4omini_json_example:
            return openai_gpt.json_example(
                house,
                model="gpt-4o-mini"
            )
        case LLMMethod.gpt_4omini_pydantic_schema:
            return openai_gpt.pydantic_schema(
                house,
                model="gpt-4o-mini"
            )
        case LLMMethod.gpt_4omini:
            return openai_gpt.function_calling(
                house,
                model="gpt-4o-mini"
            )
        case LLMMethod.gpt_4_json_example:
            return openai_gpt.json_example(
                house,
                model="gpt-3.5-turbo"
            )
        case LLMMethod.gpt_4_pydantic_schema:
            return openai_gpt.pydantic_schema(
                house,
                model="gpt-3.5-turbo"
            )
        case LLMMethod.gpt_4:
            return openai_gpt.function_calling(
                house,
                model="gpt-3.5-turbo"
            )
        case _:
            raise ValueError(f"Unknown method {method}")