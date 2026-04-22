from pathlib import Path
import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "msft_dataset.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"

START_DATE = datetime.datetime(2010, 1, 1)
END_DATE = datetime.datetime(2020, 1, 1)

RETURN_PERIOD = 5
TEST_SIZE = 0.2
SEQ_LEN = 5

ARIMA_ORDER = (1, 0, 0)

RANDOM_STATE = 42