from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

TIMEOUT = 10 

HEADERS = {
    "User-Agent": "MP3_APIFetcher_Lagarde_Vincent/1.0",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

URLS = [
    "https://world.openfoodfacts.org/api/v2/search",
    "https://world.openbeautyfacts.org/api/v2/search",
    "https://world.openproductfacts.org/api/v2/search"
]

MAX_WORKERS = 5


DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SUMMARY_FILE = DATA_DIR / "summary.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "fetcher.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

