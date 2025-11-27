import time
import json
import logging
import requests
from requests.exceptions import Timeout, ConnectionError, HTTPError
from typing import List, Dict
import pandas as pd

from core.config import TIMEOUT, HEADERS, RAW_DATA_DIR, LOG_PATH, LOG_FORMAT, URLS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH)
    formatter = logging.Formatter(LOG_FORMAT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def fetch(url, params=None) -> Dict:
    start = time.perf_counter()
    try:
        response = requests.get(url, timeout=TIMEOUT, headers=HEADERS, params=params)
        elapsed = round(time.perf_counter() - start, 3)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").split(";")[0].lower()

        if "application/json" in content_type:
            try:
                payload = response.json()
            except ValueError:
                payload = response.text[:500]
        else:
            payload = response.text[:500]

        return {
            "url": url,
            "status": response.status_code,
            "elapsed": elapsed,
            "content_type": content_type,
            "payload": payload
        }

    except Timeout:
        return {"url": url, "status": "error", "error_type": "timeout",
                "elapsed": round(time.perf_counter() - start, 3),
                "content_type": None, "payload": None}
    except ConnectionError:
        return {"url": url, "status": "error", "error_type": "connection_error",
                "elapsed": round(time.perf_counter() - start, 3),
                "content_type": None, "payload": None}
    except HTTPError as e:
        return {"url": url, "status": e.response.status_code, "error_type": "http_error",
                "elapsed": round(time.perf_counter() - start, 3),
                "content_type": e.response.headers.get("Content-Type", None),
                "payload": None}
    

def save_response(api_name: str, data: Dict):
    path = RAW_DATA_DIR / f"{api_name}_response.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Réponse sauvegardée: {path}")


def standardize_product(obj: dict, source: str) -> Dict:
    return {
        "id": obj.get("id") or obj.get("code") or obj.get("_id"),
        "title": obj.get("product_name") or obj.get("generic_name") or obj.get("brands") or "Unknown Product",
        "text": ((obj.get("ingredients_text") or "") + " " + (obj.get("labels") or "")).strip(),
        "category": obj.get("categories_tags", []),
        "source": source,
        "published_at": obj.get("created_t"),
    }

def fetch_to_dataframe(url: str, api_name: str) -> pd.DataFrame:
    resp = fetch(url)
    save_response(f"{api_name}_raw", resp)

    products = []
    if resp.get("payload") and "products" in resp["payload"]:
        products = [standardize_product(p, api_name) for p in resp["payload"]["products"]]

    df = pd.DataFrame(products)
    return df

def fetch_openfoodfacts() -> pd.DataFrame:
    return fetch_to_dataframe(URLS[0], "openfoodfacts")

def fetch_openbeautyfacts() -> pd.DataFrame:
    return fetch_to_dataframe(URLS[1], "openbeautyfacts")

def fetch_openproductfacts() -> pd.DataFrame:
    return fetch_to_dataframe(URLS[2], "openproductfacts")

def fetch_all() -> Dict[str, pd.DataFrame]:
    logger.info("Fetching OpenFoodFacts...")
    df_food = fetch_openfoodfacts()

    logger.info("Fetching OpenBeautyFacts...")
    df_beauty = fetch_openbeautyfacts()

    logger.info("Fetching OpenProductFacts...")
    df_product = fetch_openproductfacts()

    logger.info(f"Total produits récupérés : {len(df_food) + len(df_beauty) + len(df_product)}")

    return {
        "openfoodfacts": df_food,
        "openbeautyfacts": df_beauty,
        "openproductfacts": df_product
    }

if __name__ == "__main__":
    print("Test fetcher des trois APIs...")
    all_dfs = fetch_all()

    for name, df in all_dfs.items():
        print(f"\n{name} ({len(df)} produits)")
        print(df.head())
