from pydantic import BaseModel
from typing import Optional

from typing import Type

def test_housing_usecase():
    unstructured_data = "A lovely house with a garden and parking place. It is suitable for a couple and has a balcony."
    response_class = lmm_extraction_MOCK(pydantic_class=HouseFeatures, unstructured_data=unstructured_data, model_name="gpt-4o-mini", API_key="mysecretkey")
    # check if response class is actually of the same schema as class given
    assert isinstance(response_class, HouseFeatures)
    # check if llm extraction is accurate
    assert response_class.is_pet_friendly.was_extracted == False
    assert response_class.has_garden.was_extracted == True
    assert response_class.has_parking_place.was_extracted == True
    assert response_class.is_suitable_for_couple.was_extracted == True
    assert response_class.has_balcony.was_extracted == True

class FeatureInfo(BaseModel):
    was_extracted: bool
    quote: Optional[str] = None
    value: Optional[bool] = None

class HouseFeatures(BaseModel):
    is_pet_friendly: FeatureInfo
    has_garden: FeatureInfo
    has_parking_place: FeatureInfo
    is_suitable_for_couple: FeatureInfo
    has_balcony: FeatureInfo

def lmm_extraction_MOCK(pydantic_class:Type[BaseModel], unstructured_data:str, model_name:str, API_key:str):
    """
    This is a mock class to help visualize how I think the function i will be testing in the future will work
    using this since I don't actually have anything to test yet
    """
    ipf = FeatureInfo(was_extracted=False)
    hg = FeatureInfo(was_extracted=True)
    hpp = FeatureInfo(was_extracted=True)
    isfc = FeatureInfo(was_extracted=True)
    hb = FeatureInfo(was_extracted=True)
    return HouseFeatures(is_pet_friendly=ipf, has_garden=hg, has_parking_place=hpp, is_suitable_for_couple=isfc, has_balcony=hb)
