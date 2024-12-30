import json
from house_bot.enrichment.enrichment_types import House, HouseFeatures


import mlflow
import mlflow.llm
from openai import OpenAI
import openai


def json_example(
    house: House, model: str, deployment_id: str
) -> HouseFeatures:
    prompt = """
    A house listing will follow, as it appeared on a housing website. Please use the template to extract information from the listing in a JSON response.

    Return nothing else except a valid JSON object. Do not surround the JSON object with backticks or any other text.

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
    prompt += f"""

    # house listing
    {str(house)}
    """

    context = "Assistant is a large language model designed to extract structured data from text."
    input_message = {
        "role": "system",
        "content": context,
    }
    prompt_message = {"role": "user", "content": prompt}
    messages = [
        input_message,
        prompt_message,
    ]
    hyperparams = {
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "max_tokens": None,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    response = openai.ChatCompletion.create(
        model=model,
        deployment_id=deployment_id,
        messages=messages,
        **hyperparams,
    )
    response_text = response.choices[0]["message"]["content"]

    # optionally log to MLflow
    is_run_active = mlflow.active_run is not None
    if is_run_active:
        outputs = [response]
        mlflow.log_params(hyperparams)
        mlflow.llm.log_predictions(
            inputs=[input_message], outputs=outputs, prompts=[prompt_message]
        )

    # construct HouseFeatures object from text
    house_features = HouseFeatures.model_validate_json(response_text)

    return house_features


def pydantic_schema(
    house: House, model: str, deployment_id: str
) -> HouseFeatures:
    output_template = HouseFeatures.model_json_schema()
    output_template_str = json.dumps(output_template)
    prompt = f"""
    A house listing will follow. Your task is to extract details from the house listing. Fill in the following template:

    Return nothing else except a valid JSON object. Do not surround the JSON object with backticks or any other text.

    # template
    {output_template_str}

    # house listing
    {str(house)}
    """

    context = "Assistant is a large language model designed to extract structured data from text."
    input_message = {
        "role": "system",
        "content": context,
    }
    prompt_message = {"role": "user", "content": prompt}
    messages = [
        input_message,
        prompt_message,
    ]
    hyperparams = {
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "max_tokens": None,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    response = openai.ChatCompletion.create(
        model=model,
        deployment_id=deployment_id,
        messages=messages,
        **hyperparams,
    )
    response_text = response.choices[0]["message"]["content"]

    # optionally log to MLflow
    is_run_active = mlflow.active_run is not None
    if is_run_active:
        outputs = [response]
        mlflow.log_params(hyperparams)
        mlflow.llm.log_predictions(
            inputs=[input_message], outputs=outputs, prompts=[prompt_message]
        )

    # construct HouseFeatures object from text
    house_features = HouseFeatures.model_validate_json(response_text)

    return house_features


def function_calling(
    house: House, model: str
) -> HouseFeatures:
    """
    Extract structured features from a housing listing using a language model.

    This function takes an unstructured text representation of a house and
    converts it into a structured format defined by the HouseFeatures class. 
    Optionally, the function logs the parameters and predictions to MLflow if an active run exists.

    Parameters:
    ----------
    house : House
        An instance of the House class containing unstructured listing information.
    model : str
        The identifier of the OpenAI language model to be used.

    Returns:
    -------
    HouseFeatures
        An instance of HouseFeatures containing the extracted structured data.

    Notes:
    -----
    - The function has been modified to use the Structured Outputs OpenAI's chat completions to ensure 
    the responses adhere to the required JSON schema.
    """
    prompt = str(house)

    # initialize client
    client = OpenAI()
    tools = [openai.pydantic_function_tool(HouseFeatures)]

    # construct the messages for the model
    # uses context recommended in documentation
    context = "You are an expert at structured data extraction. You will be given unstructured text from a housing listing and should convert it into the given structure."
    input_message = {
        "role": "system",
        "content": context,
    }
    prompt_message = {"role": "user", "content": prompt}
    messages = [
        input_message,
        prompt_message,
    ]

    hyperparams = {
        "temperature": 0.5,
        "top_p": 1.0,
        "n": 1,
        "max_tokens": None,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }

    # call the model
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        **hyperparams
    )

    # extract structured features from the response
    house_features = HouseFeatures.from_response(response)
    # to see the json message check response.choices[0].message.tool_calls[0].function
    print(response.choices[0].message.tool_calls[0].function) # TODO: if debug

    # TODO: discuss how to resolve refusals!!!
    # if response.choices[0].message.refusal is not None:
    #     return ...
    # else:
    #     return <as it is>

    # optionally log to MLflow
    is_run_active = mlflow.active_run is not None
    if is_run_active:
        outputs = [response]
        mlflow.log_params(hyperparams)
        mlflow.llm.log_predictions(
            inputs=[input_message], outputs=outputs, prompts=[prompt_message]
        )

    return house_features
