# src/read_data.py

import pandas as pd
import pickle
import logging
from pathlib import Path
from urllib.error import URLError
from ucimlrepo import fetch_ucirepo
from typing import Tuple

# --- Constants ---
DATASET_ID = 296
CACHE_DIR = Path("data/cache")
CACHE_FILE = CACHE_DIR / "diabetes_dataset.pkl"

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(use_cache: bool = True, verbose: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fetches the "Diabetes 130-US hospitals" dataset from the UCI repository.

    This function includes file-based caching to avoid re-downloading the data
    on every run. It loads the data from a local cache if available, otherwise
    it fetches from the remote repository and saves it to the cache.

    Args:
        use_cache (bool): If True, attempts to load from and save to the local cache.
        verbose (bool): If True, prints the dataset's metadata and variable info.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the feature DataFrame (X)
                                        and the target Series (y).
                                        
    Raises:
        URLError: If fetching from the UCI repository fails due to network issues.
    """
    # 1. Attempt to load from cache
    if use_cache and CACHE_FILE.exists():
        logging.info(f"Loading data from cache: {CACHE_FILE}")
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
            return data['X'], data['y']
        except (pickle.UnpicklingError, KeyError, EOFError) as e:
            logging.warning(f"Cache file is corrupted: {e}. Refetching data.")

    # 2. Fetch from UCI repository if cache is not used or fails
    logging.info(f"Fetching dataset ID {DATASET_ID} from UCI repository...")
    try:
        repo = fetch_ucirepo(id=DATASET_ID)
        X = repo.data.features
        y = repo.data.targets
    except URLError as e:
        logging.error(f"Failed to fetch data from UCI repository. Check your internet connection. Error: {e}")
        raise

    # 3. Save to cache for future use
    if use_cache:
        logging.info(f"Saving data to cache: {CACHE_FILE}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump({'X': X, 'y': y}, f)
        except IOError as e:
            logging.error(f"Could not write to cache file at {CACHE_FILE}. Error: {e}")

    # 4. Optional verbose output
    if verbose:
        logging.info(f"--- Dataset Metadata ---\n{repo.metadata}\n")
        logging.info(f"--- Variable Information ---\n{repo.variables}\n")

    return X, y

if __name__ == '__main__':
    # This block allows you to run the script directly to test it
    print("Running data loader test...")
    
    # First run (should fetch and cache)
    print("\n--- First run (fetching and caching) ---")
    features, target = load_data(verbose=True)
    print("\nFeatures (X) shape:", features.shape)
    print("Target (y) shape:", target.shape)
    print("Data loaded successfully.")
    
    # Second run (should load from cache)
    print("\n--- Second run (loading from cache) ---")
    features_cached, target_cached = load_data()
    print("\nFeatures (X) shape:", features_cached.shape)
    print("Target (y) shape:", target_cached.shape)
    print("Data loaded successfully from cache.")
