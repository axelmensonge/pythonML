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
    "https://world.openpetfoodfacts.org/api/v2/search"
]

MAX_WORKERS = 5

REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SUMMARY_FILE = REPORTS_DIR / "summary.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "marketing.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

