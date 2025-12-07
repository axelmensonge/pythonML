import time
import json
import logging
import requests
from requests.exceptions import Timeout, ConnectionError, HTTPError
from typing import Dict
from core.config import SUMMARY_FILE

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
            "url": response.url,
            "status": response.status_code,
            "elapsed": elapsed,
            "content_type": content_type,
            "payload": payload
        }

    except Timeout:
        return {"url": url, "status": "error", "error_type": "timeout",
                "elapsed": round(time.perf_counter() - start, 3), "content_type": None, "payload": None}

    except ConnectionError:
        return {"url": url, "status": "error", "error_type": "connection_error",
                "elapsed": round(time.perf_counter() - start, 3), "content_type": None, "payload": None}

    except HTTPError as e:
        return {"url": url, "status": e.response.status_code, "error_type": "http_error",
                "elapsed": round(time.perf_counter() - start, 3),
                "content_type": e.response.headers.get("Content-Type", None), "payload": None}


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
    }



def save_summary_entry(api_name: str, status: int, elapsed: float, total_products: int):
    summary = {
        "api": api_name,
        "status": status,
        "response_time_seconds": elapsed,
        "total_products_saved": total_products
    }

    if SUMMARY_FILE.exists():
        try:
            with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except:
            existing = {}
    else:
        existing = {}

    existing[api_name] = summary

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    logger.info(f"Résumé mis à jour dans : {SUMMARY_FILE}")



def fetch_all_pages(url: str, api_name: str, max_products: int = 1000) -> pd.DataFrame:
    all_products = []
    page = 1
    page_size = 100

    total_elapsed = 0
    last_status = None

    while len(all_products) < max_products:
        params = {
            "action": "process",
            "json": "true",
            "page": page,
            "page_size": page_size
        }

        resp = fetch(url, params=params)

        last_status = resp.get("status")
        total_elapsed += resp.get("elapsed", 0)

        if resp.get("status") == "error" or not resp.get("payload"):
            logger.warning(f"Erreur à la page {page} pour {api_name}")
            break

        products = resp["payload"].get("products", [])
        if not products:
            logger.info(f"Plus de produits à la page {page}")
            break

        for p in products:
            all_products.append(standardize_product(p, api_name))
            if len(all_products) >= max_products:
                break

        page += 1
        time.sleep(0.2)

    df = pd.DataFrame(all_products)

    save_response(f"{api_name}_all", {"products": all_products, "total": len(all_products)})

    save_summary_entry(api_name, last_status, total_elapsed, len(all_products))

    return df



def fetch_openfoodfacts() -> pd.DataFrame:
    return fetch_all_pages(URLS[0], "openfoodfacts")


def fetch_openbeautyfacts() -> pd.DataFrame:
    return fetch_all_pages(URLS[1], "openbeautyfacts")


def fetch_openpetfoodfacts() -> pd.DataFrame:
    return fetch_all_pages(URLS[2], "openpetfoodfacts")


def fetch_all() -> Dict[str, pd.DataFrame]:
    logger.info("Recheche pour OpenFoodFacts...")
    df_food = fetch_openfoodfacts()

    logger.info("Recheche pour OpenBeautyFacts...")
    df_beauty = fetch_openbeautyfacts()

    logger.info("Recheche pour OpenPetFoodFacts...")
    df_pet = fetch_openpetfoodfacts()

    total = len(df_food) + len(df_beauty) + len(df_pet)
    logger.info(f"Total produits récupérés : {total}")

    return {
        "openfoodfacts": df_food,
        "openbeautyfacts": df_beauty,
        "openpetfoodfacts": df_pet
    }


if __name__ == "__main__":
    pd.set_option("display.max_rows", 1000)
    print("Récupérations des produits pour les trois APIs")
    all_dfs = fetch_all()

    for name, df in all_dfs.items():
        print(f"\n{name} ({len(df)} produits)")
        print(df.head(1000))