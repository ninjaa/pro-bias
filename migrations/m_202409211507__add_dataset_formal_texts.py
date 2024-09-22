import logging
from pathlib import Path

from src.config import config
from weave import Dataset
import weave
import pandas as pd

log = logging.getLogger(__name__)

#
# Migration
#


FILE_PATH = Path(__file__).parent.parent / "datasets" / "formal_texts.csv"


def up():
    weave.init(config.WEAVE_PROJECT)

    df = pd.read_csv(FILE_PATH)
    dataset_rows = df.to_dict('records')

    dataset = Dataset(name='formal_texts', rows=dataset_rows)
    weave.publish(dataset)

    log.info(f"Uploaded {len(dataset_rows)} items to formal_texts dataset")


if __name__ == "__main__":
    up()
