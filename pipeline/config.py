"""
Configuration: models, paths, item constants.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

FILING_FOLDERS = {
    "folder_1": DATA_DIR / "folder_1",
    "folder_2": DATA_DIR / "folder_2",
    "folder_3": DATA_DIR / "folder_3",
}

GT_DIRS = {
    "ground_truth_1": DATA_DIR / "ground_truth_1",
    "ground_truth_2": DATA_DIR / "ground_truth_2",
    "ground_truth_3": DATA_DIR / "ground_truth_3",
}

# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
LLM_TIMEOUT_SECONDS = 120
MAX_RETRIES = 2
DEFAULT_CONCURRENCY = 3
DEFAULT_BUDGET = 5.00

# Sequential order of items in a 10-K filing (22 items + signatures)
ITEM_SEQ_ORDER: list[str] = [
    "item1", "item1a", "item1b",
    "item2", "item3", "item4",
    "item5", "item6", "item7", "item7a",
    "item8", "item9", "item9a", "item9b", "item9c",
    "item10", "item11", "item12", "item13", "item14",
    "item15", "item16",
    "crossReference",
    "signatures",
]

# Reverse lookup: item name -> canonical index
ITEM_SEQ_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(ITEM_SEQ_ORDER)
}

# Items that should almost always appear in a well-formed 10-K
# NOTE: signatures excluded — many GT files don't include it as a separate item,
# so forcing it causes false positives that hurt doc retrieval.
EXPECTED_ITEMS: frozenset[str] = frozenset([
    "item1", "item1a", "item7", "item8", "item15",
])
