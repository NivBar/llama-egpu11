import pandas as pd
import json
from pyserini.search import FaissSearcher

searcher = FaissSearcher(
    './pyserini_output',
    'facebook/dpr-question_encoder-multiset-base'
)


def create_jsonl_file():
    df = pd.read_csv("/lv_local/home/niv.b/llama/greg_data.csv")
    filename = '/lv_local/home/niv.b/llama/greg_output.jsonl'
    with open(filename, 'w') as file:
        for _, row in df.iterrows():
            json_object = {
                "id": row['docno'],
                "contents": row['current_document'].replace('\n', ' ').strip() + '\n'
            }
            file.write(json.dumps(json_object) + '\n')


def retrieve_top_docs(query, k=10):
    global searcher
    hits = searcher.search(query, k=k)
    return [hits[i].docid for i in range(k)]