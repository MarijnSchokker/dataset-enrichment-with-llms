from house_bot.enrichment.enrichment_types import House, HouseFeatures


import mlflow
import mlflow.llm
from vertexai.language_models import ChatModel


import json

def json_schema(house: House) -> HouseFeatures:
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    hyperparams = {
        "max_output_tokens": 1024,
        "temperature": 0.0,
        "top_p": 0.8,
        "top_k": 40,
    }

    prompt_message = """
    A house listing will follow, as it appeared on a housing website. Please use the template to extract information from the listing in a JSON response:

    # template
    ```
    {
        "is_pet_friendly": {
            "was_extracted": ..., # type: boolean (false | true)
            "quote": ..., # type: str (str | null)
            "value": ..., # type: bool (false | true | null)
        },
        "has_garden": {
            "was_extracted": ..., # type: boolean (false | true)
            "quote": ..., # type: str (str | null)
            "value": ..., # type: bool (false | true | null)
        },
        "has_parking_place": {
            "was_extracted": ..., # type: boolean (false | true)
            "quote": ..., # type: str (str | null)
            "value": ..., # type: bool (false | true | null)
        },
        "is_suitable_for_couple": {
            "was_extracted": ..., # type: boolean (false | true)
            "quote": ..., # type: str (str | null)
            "value": ..., # type: bool (false | true | null)
        },
        "has_balcony": {
            "was_extracted": ..., # type: boolean (false | true)
            "quote": ..., # type: str (str | null)
            "value": ..., # type: bool (false | true | null)
        },
    }
    """
    prompt_message += f"""

    # house listing
    {str(house)}
    """
    context = "Assistant is a large language model designed to extract structured data from text."

    chat = chat_model.start_chat(
        context=context,
    )
    response = chat.send_message(prompt_message, **hyperparams)
    response_str = response.text

    # optionally log to MLflow
    is_run_active = mlflow.active_run is not None
    if is_run_active:
        outputs = [response_str]
        mlflow.log_params(hyperparams)
        mlflow.llm.log_predictions(
            inputs=[context], outputs=outputs, prompts=[prompt_message]
        )

    # construct HouseFeatures object from text
    house_features = HouseFeatures.model_validate_json(response.text)

    return house_features

def pydantic_schema(house: House) -> HouseFeatures:
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    hyperparams = {
        "max_output_tokens": 1024,
        "temperature": 0.0,
        "top_p": 0.8,
        "top_k": 40,
    }

    output_template = HouseFeatures.model_json_schema()
    output_template_str = json.dumps(output_template)
    prompt_message = f"""
    A house listing will follow. Your task is to extract details from the house listing. Fill in the following template:

    # template
    {output_template_str}
    
    # house listing
    {str(house)}
    """
    context = "Assistant is a large language model designed to extract structured data from text."

    chat = chat_model.start_chat(
        context=context,
    )
    response = chat.send_message(prompt_message, **hyperparams)
    response_str = response.text

    # optionally log to MLflow
    is_run_active = mlflow.active_run is not None
    if is_run_active:
        outputs = [response_str]
        mlflow.log_params(hyperparams)
        mlflow.llm.log_predictions(
            inputs=[context], outputs=outputs, prompts=[prompt_message]
        )

    # construct HouseFeatures object from text
    house_features = HouseFeatures.model_validate_json(response.text)

    return house_features