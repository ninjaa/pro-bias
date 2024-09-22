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
# inspired by https://www.reddit.com/r/Screenwriting/comments/smpofr/free_movie_ideas_megapost_2_january_2022/
# https://claude.ai/chat/7e9aeb75-c605-42f2-9982-5282113a86f5

FILE_PATH = Path(__file__).parent.parent / "datasets" / "sonnet_movie_ideas.csv"

def up():
    weave.init(config.WEAVE_PROJECT)

    df = pd.read_csv(FILE_PATH)
    dataset_rows = df.to_dict('records')

    dataset = Dataset(name='sonnet_movie_ideas', rows=dataset_rows)
    weave.publish(dataset)

    log.info(f"Uploaded {len(dataset_rows)} items to sonnet_movie_ideas dataset")

if __name__ == "__main__":
    up()