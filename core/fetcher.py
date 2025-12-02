import time
import json
import requests
from requests.exceptions import Timeout, ConnectionError, HTTPError
from typing import Dict
import pandas as pd
from core.logger import get_logger

logger = get_logger(__name__)


class Fetcher:
    def __init__(self, timeout, max_products, page_size, page, headers, urls, raw_data_dir):
        self.timeout = timeout
        self.headers = headers
        self.urls = urls
        self.max_products = max_products
        self.page_size = page_size
        self.page = page
        self.raw_data_dir = raw_data_dir
        logger.info("Fetcher initialisé")


    @staticmethod
    def standardize_product(obj: dict, source: str) -> Dict:
        return {
            "id": obj.get("id") or obj.get("code") or obj.get("_id"),
            "title": obj.get("product_name") or obj.get("generic_name") or obj.get("brands") or "Unknown Product",
            "text": ((obj.get("ingredients_text") or "") + " " + (obj.get("labels") or "")).strip(),
            "category": obj.get("categories_tags", []),
            "source": source,
        }


    def save_response(self, api_name: str, data: Dict):
        path = self.raw_data_dir / f"{api_name}_response.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Réponse sauvegardée: {path}")


    def fetch(self, url, params) -> Dict:
        start = time.perf_counter()
        try:
            response = requests.get(url, timeout=self.timeout, headers=self.headers, params=params)
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
            return {"url": url, "status": "error", "error_type": "timeout", "elapsed": round(time.perf_counter()-start,3), "content_type": None, "payload": None}

        except ConnectionError:
            return {"url": url, "status": "error", "error_type": "connection_error", "elapsed": round(time.perf_counter()-start,3), "content_type": None, "payload": None}

        except HTTPError as e:
            return {"url": url, "status": e.response.status_code, "error_type": "http_error", "elapsed": round(time.perf_counter()-start,3), "content_type": e.response.headers.get("Content-Type", None), "payload": None}


    def fetch_all_pages(self, url: str, api_name: str, max_products: int = 1000) -> pd.DataFrame:
        all_products = []
        page = self.page
        page_size = self.page_size

        while len(all_products) < max_products:
            params = {
                "action": "process",
                "json": "true",
                "page": page,
                "page_size": page_size
            }

            resp = self.fetch(url, params)

            if resp.get("status") == "error" or not resp.get("payload"):
                logger.warning(f"Erreur ou fin des produits à la page {page} pour {api_name}")
                break

            products = resp["payload"].get("products", [])
            if not products:
                logger.info(f"Plus de produits à la page {page} pour {api_name}")
                break

            for p in products:
                all_products.append(self.standardize_product(p, api_name))
                if len(all_products) >= max_products:
                    break

            logger.info(f"{api_name}: Page {page} récupérée ({len(products)} produits)")
            page += 1
            time.sleep(0.2)

        df = pd.DataFrame(all_products)
        self.save_response(f"{api_name}_all", {"total_products": len(all_products)})
        return df


    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        logger.info("Recheche pour OpenFoodFacts...")
        df_food = self.fetch_all_pages(self.urls[0], "openfoodfacts", self.max_products)

        logger.info("Recheche pour OpenBeautyFacts...")
        df_beauty = self.fetch_all_pages(self.urls[1], "openbeautyfacts", self.max_products)

        logger.info("Recheche pour OpenPetFoodFacts...")
        df_pet = self.fetch_all_pages(self.urls[2], "openpetfoodfacts", self.max_products)

        total = len(df_food) + len(df_beauty) + len(df_pet)
        logger.info(f"Total produits récupérés : {total}")

        return {
            "openfoodfacts": df_food,
            "openbeautyfacts": df_beauty,
            "openpetfoodfacts": df_pet
        }