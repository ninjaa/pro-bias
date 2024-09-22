import logging
from pathlib import Path

from weave import Dataset
import weave
import pandas as pd

log = logging.getLogger(__name__)

#
# Migration
#
# inspired by https://www.reddit.com/r/Screenwriting/comments/smpofr/free_movie_ideas_megapost_2_january_2022/
# https://chatgpt.com/share/66ef86f6-3424-800b-bf98-9ad4039f2dbf

FILE_PATH = Path(__file__).parent.parent / "datasets" / \
    "o1_preview_movie_ideas.csv"


def up():
    weave.init('pro-bias')

    df = pd.read_csv(FILE_PATH)
    dataset_rows = df.to_dict('records')

    dataset = Dataset(name='o1_preview_movie_ideas', rows=dataset_rows)
    weave.publish(dataset)

    log.info(
        f"Uploaded {len(dataset_rows)} items to o1_preview_movie_ideas dataset")


if __name__ == "__main__":
    up()
