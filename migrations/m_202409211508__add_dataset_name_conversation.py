import logging
from pathlib import Path

from weave import Dataset
import weave
import json

log = logging.getLogger(__name__)

#
# Migration
#

FILE_PATH = Path(__file__).parent.parent / "datasets" / "sonnet_chats.json"

def up():
    weave.init('pro-bias')

    with open(FILE_PATH, 'r') as file:
        data = json.load(file)
    
    dataset_rows = []
    for chat in data['chats']:
        conversation_text = json.dumps(chat)
        dataset_rows.append({
            'conversation_text': conversation_text,
            'expected_output': 'UNUSED: Short name for the conversation'
        })

    dataset = Dataset(name='name_conversation', rows=dataset_rows)
    weave.publish(dataset)

    log.info(f"Uploaded {len(dataset_rows)} items to name_conversation dataset")

if __name__ == "__main__":
    up()