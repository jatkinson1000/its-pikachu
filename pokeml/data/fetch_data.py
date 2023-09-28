"""Module for handling kaggle datasets."""
import kaggle

# Authenticate access to the kaggle API using environment variables
kaggle.api.authenticate()


def download_kaggle_dataset(
    dataset_name: str,
    location: str,
    extract: bool = True,
) -> None:
    """
    Downloads a dataset from kaggle using the API to a predefined location.

    Parameters
    ----------
    dataset_name : str
        name of a kaggle dataset "<USER>/<DATASET_NAME>"
    location : str
        string defining the path to save the data at
    extract : bool
        unzip the files once downloaded?
    """
    kaggle.api.dataset_download_files(dataset_name, path=location, unzip=extract)
