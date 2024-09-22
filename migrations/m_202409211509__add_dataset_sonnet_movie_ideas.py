import logging
from pathlib import Path

from weave import Dataset
import weave
import pandas as pd

log = logging.getLogger(__name__)

#
# Migration
#

FILE_PATH = Path(__file__).parent.parent / "datasets" / "sonnet_movie_ideas.csv"

def up():
    weave.init('pro-bias')

    df = pd.read_csv(FILE_PATH)
    dataset_rows = df.to_dict('records')

    dataset = Dataset(name='sonnet_movie_ideas', rows=dataset_rows)
    weave.publish(dataset)

    log.info(f"Uploaded {len(dataset_rows)} items to sonnet_movie_ideas dataset")

if __name__ == "__main__":
    up()