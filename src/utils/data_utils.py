import csv
import json
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split


def flatten_json(nested_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens a nested JSON object into a single-level dictionary.

    Args:
        nested_json (Dict[str, Any]): The nested JSON data to flatten.

    Returns:
        Dict[str, Any]: A dictionary with flattened key-value pairs.
    """
    flat_dict: Dict[str, Any] = {}

    def flatten(data: Any, parent_key: str = "") -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{parent_key}_{key}" if parent_key else key
                flatten(value, new_key)
        elif isinstance(data, list):
            for i, value in enumerate(data):
                new_key = f"{parent_key}_{i}" if parent_key else str(i)
                flatten(value, new_key)
        else:
            flat_dict[parent_key] = data

    flatten(nested_json)
    return flat_dict


def json_to_csv(json_path: str, csv_path: str) -> None:
    """
    Reads a JSON file, flattens its contents, and writes the data to a CSV file.

    Args:
        json_path (str): Path to the JSON file containing reviews.
        csv_path (str): Path where the output CSV file should be saved.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        json_data: List[Dict[str, Any]] = json.load(f)

    flattened_data: List[Dict[str, Any]] = [flatten_json(review) for review in json_data]

    headers = {key for review in flattened_data for key in review.keys()}

    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(headers), delimiter="\t")
        writer.writeheader()
        writer.writerows(flattened_data)

    print(f"Data has been successfully written to {csv_path}")


def create_balanced_test_set(csv_path: str, rating_column: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a balanced train and test set by selecting a fixed percentage of each rating class.
    Ensures that no NaN values are present in the dataset.

    Args:
        csv_path (str): Path to the CSV file containing data.
        rating_column (str): Column name that holds rating values.
        test_size (float): Proportion of each class to be included in the test set (default is 20%).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The train and test datasets as pandas DataFrames.
    """
    df = pd.read_csv(csv_path, sep="\t")

    if rating_column not in df.columns:
        raise ValueError(f"Column '{rating_column}' not found in dataset.")

    test_sets: List[pd.DataFrame] = []
    train_sets: List[pd.DataFrame] = []

    for rating in df[rating_column].unique():
        class_subset = df[df[rating_column] == rating]

        if len(class_subset) >= 5:  # Ensure sufficient members per class for splitting
            train_subset, test_subset = train_test_split(
                class_subset,
                test_size=test_size,
                random_state=42,
                stratify=class_subset[[rating_column]],  # Correct stratification by rating column
            )
            train_sets.append(train_subset)
            test_sets.append(test_subset)

    train_set = pd.concat(train_sets).reset_index(drop=True)
    test_set = pd.concat(test_sets).reset_index(drop=True)

    return train_set, test_set


if __name__ == "__main__":
    # Convert JSON data to CSV
    json_to_csv("../data/reviews.json", "data/reviews.csv")

    # Create balanced train and test sets
    train_set, test_set = create_balanced_test_set("data/reviews.csv", "rating")

    # Save the train and test sets to CSV files
    train_set.to_csv("data/balanced_train_set.csv", index=False, sep="\t")
    test_set.to_csv("data/balanced_test_set.csv", index=False, sep="\t")

    print("Balanced train and test sets have been created and saved.")
