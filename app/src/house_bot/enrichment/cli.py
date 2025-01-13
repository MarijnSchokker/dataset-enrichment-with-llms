from pathlib import Path

import mlflow
import mlflow.llm
import typer
from house_bot.enrichment.enrichment_strategies import enrich_house_with_llm
from house_bot.enrichment.enrichment_types import CACHE_FOLDER_NAME, HOUSES_CACHE, House, HouseFeature, HouseFeatures, LLMMethod
from house_bot.enrichment.housing_data_io import fetch_house_from_disk, fetch_housing_data, load_house_features, save_house_features
import pandas as pd

# TODO: Differences with scrapper:  3 additional functions that were interacting with firestore 
# add_house_description, find_and_add_house_description, find_and_add_house_descriptions


app = typer.Typer()

HOUSE_EXTRACTED_FEATURES_CACHE = Path(CACHE_FOLDER_NAME) / "house_extracted_features"


def attach_house_features(house: House, house_features: HouseFeatures) -> House:
    for feature_name in house_features.model_fields:
        feature: HouseFeature = getattr(house_features, feature_name)
        setattr(house, feature_name, feature)

    return house


@app.command()
def save_houses_to_disk(cache_file: Path = HOUSES_CACHE):
    houses = fetch_housing_data(cache_file, ignore_cache=True)
    print(f"Saving {len(houses)} houses to {cache_file}")


@app.command()
def enrich_house(
    house_id: str,
    method: LLMMethod,
    cache_file: Path = HOUSES_CACHE,
):
    # retrieve house as House object
    house = fetch_house_from_disk(house_id=house_id, cache_file=cache_file)

    # enrich house
    house_features = enrich_house_with_llm(house, method=method)

    # save house features to json
    house_features_json_path = (
        HOUSE_EXTRACTED_FEATURES_CACHE / method.name / f"{house_id}.json"
    )
    save_house_features(house_features, json_path=house_features_json_path)


def compare_house_features(predicted: HouseFeatures, actual: HouseFeatures) -> dict:
    # convert to dataframes
    predicted_df = pd.DataFrame(predicted.model_dump())
    predicted_df = predicted_df.loc[["was_extracted", "value"]]
    actual_df = pd.DataFrame(actual.model_dump())
    actual_df = actual_df.loc[["was_extracted", "value"]]

    # compute correct over total
    correct_entries = (predicted_df == actual_df).sum().sum()
    total_entries = len(predicted_df.columns) * len(predicted_df.index)
    percentage_correct = correct_entries / total_entries

    # compute correct per category
    correct_per_category = (predicted_df == actual_df).sum(axis=1)
    total_per_category = len(predicted_df.columns)
    percentage_correct_per_category = correct_per_category / total_per_category
    correct_per_category = correct_per_category.add_prefix("number_correct_")
    percentage_correct_per_category = percentage_correct_per_category.add_prefix(
        "percentage_correct_"
    )

    # convert to dict
    metrics = {
        "percentage_correct": percentage_correct,
        "number_correct": correct_entries,
        "total": total_entries,
        **correct_per_category.to_dict(),
        **percentage_correct_per_category.to_dict(),
    }

    return metrics


def compare_house_features_str(predicted: HouseFeatures, actual: HouseFeatures) -> str:
    predicted_df = pd.DataFrame(predicted.model_dump())
    predicted_df = predicted_df.loc[["was_extracted", "value"]]

    actual_df = pd.DataFrame(actual.model_dump())
    actual_df = actual_df.loc[["was_extracted", "value"]]

    comparison = predicted_df.compare(
        actual_df,
        result_names=("extracted (LLM)", "actual"),
        keep_shape=True,
        keep_equal=True,
    )
    comparison_html = comparison.to_html()
    return comparison_html


def compare_house_features_print_colourful_diff(predicted: HouseFeatures, actual: HouseFeatures) -> str:
    predicted_df = pd.DataFrame(predicted.model_dump())
    predicted_df = predicted_df.loc[["was_extracted", "value"]]

    actual_df = pd.DataFrame(actual.model_dump())
    actual_df = actual_df.loc[["was_extracted", "value"]]

    def combine_values(predicted_value, actual_value):
        green_color = "#e6ffe6"
        red_color = "#ffe6e6"
        if predicted_value == actual_value:
            return predicted_value
        else:
            return (
                f"<del style='background:{red_color}'>{predicted_value}</del>"
                f"<ins style='background:{green_color}'>{actual_value}</ins>"
            )

    def combine_series(predicted_series, actual_series):
        return predicted_series.combine(actual_series, func=combine_values)

    with_annotations = predicted_df.combine(actual_df, func=combine_series)

    html = with_annotations.to_html(escape=False)

    return html

def compute_and_log_enrichment_score(house_features: HouseFeatures, house_id: str):
    # load ground truth house features
    ground_truth_house_features_json_path = (
        Path("house_features_validation_data") / f"{house_id}.json"
    )
    ground_truth_house_features = load_house_features(
        json_path=ground_truth_house_features_json_path
    )

    # compare two pydantic models on their similarity
    metrics = compare_house_features(house_features, ground_truth_house_features)
    diff_table = compare_house_features_str(
        house_features, ground_truth_house_features
    )
    diff_colourful_html = compare_house_features_print_colourful_diff(
        house_features, ground_truth_house_features
    )

    # log the numbers as metrics in an MLflow run
    mlflow.log_metrics(metrics)
    mlflow.log_text(diff_table, "diff_table.html")
    mlflow.log_text(diff_colourful_html, "diff_colourful_html.html")


@app.command()
def enrichment_score(house_id: str, method: LLMMethod, cache_file: Path = HOUSES_CACHE):
    with mlflow.start_run():
        # load extracted house features
        house_features_json_path = (
            HOUSE_EXTRACTED_FEATURES_CACHE / method.name / f"{house_id}.json"
        )
        house_features = load_house_features(json_path=house_features_json_path)

        # retrieve house as House object
        house = fetch_house_from_disk(house_id=house_id, cache_file=cache_file)

        # log basic details with MLFlow
        mlflow.log_text(house.description, "house_description.txt")
        mlflow.log_param("house_id", house_id)
        mlflow.log_param("method", method.name)

        # compute and log enrichment score
        compute_and_log_enrichment_score(house_features, house_id)


@app.command()
def enrich_and_score_house(house_id: str, method: LLMMethod, cache_file: Path = HOUSES_CACHE):
    with mlflow.start_run():
        # retrieve house as House object
        house = fetch_house_from_disk(house_id=house_id, cache_file=cache_file)

        # log basic details with MLFlow
        mlflow.log_text(house.description, "house_description.txt")
        mlflow.log_param("house_id", house_id)
        mlflow.log_param("method", method.name)

        # 1) enrich house
        house_features = enrich_house_with_llm(house, method=method)

        # save house features to json
        house_features_json_path = (
            HOUSE_EXTRACTED_FEATURES_CACHE / method.name / f"{house_id}.json"
        )
        save_house_features(house_features, json_path=house_features_json_path)

        # 2) compute and log enrichment score
        compute_and_log_enrichment_score(house_features, house_id)


if __name__ == "__main__":
    app()
