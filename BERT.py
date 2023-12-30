from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET
import re
import pandas as pd

cp = 'llama2'

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model and tokenizer
# tokenizer_name = 'bert-large-uncased'
# model_name = "castorini/monobert-large-msmarco"


tokenizer_name = "amberoad/bert-multilingual-passage-reranking-msmarco"
model_name = "amberoad/bert-multilingual-passage-reranking-msmarco"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

# File paths
working_set_file_path = f'./BERT_data/working_set_{cp}.trectext'
bot_followup_file_path = f'./BERT_data/bot_followup_{cp}.trectext'
queries_file_path = f'./BERT_data/queries_{cp}.xml'


# Function to parse the working set file and create the queries_to_docnos dictionary
def parse_working_set(file_path):
    queries_to_docnos = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Working set parsing", total=len(lines)):
            parts = line.strip().split()
            if len(parts) >= 3:
                qid, _, docno = parts[:3]
                if qid not in queries_to_docnos:
                    queries_to_docnos[qid] = []
                queries_to_docnos[qid].append(docno)
    return queries_to_docnos


# Function to parse the bot followup file and create the docs_dict dictionary
def parse_bot_followup(file_path):
    docs_dict = {}
    current_docno = None
    buffer = []
    collecting = False

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Bot followup parsing:", total=len(lines)):
            if line.startswith('<DOCNO>'):
                if current_docno and buffer:
                    docs_dict[current_docno] = ' '.join(buffer).strip()
                current_docno = line.strip().replace('<DOCNO>', '').replace('</DOCNO>', '').strip()
                buffer = []
                collecting = False
            elif line.startswith('<TEXT>'):
                collecting = True
            elif line.startswith('</TEXT>'):
                collecting = False
            elif collecting:
                buffer.append(line.strip())

        # Add the last document
        if current_docno and buffer:
            docs_dict[current_docno] = ' '.join(buffer).strip()

    return docs_dict


# Function to parse the queries file and create the queries_dict dictionary
def parse_queries(file_path):
    queries_dict = {}
    tree = ET.parse(file_path)
    root = tree.getroot()
    for query in tqdm(root.findall('.//query'), desc="Queries parsing:", total=len(root.findall('.//query'))):
        query_id = query.find('number').text.strip()
        query_text = re.search(r'\((.*?)\)', query.find('text').text).group(1)
        queries_dict[query_id] = query_text
    return queries_dict


def create_rank_df(scores_dict):
    rows = []
    for query_id, docs_scores in tqdm(scores_dict.items(), desc="Creating rank df:", total=len(scores_dict)):
        for doc_no, score in docs_scores.items():
            rows.append({'query_id': query_id, 'doc_no': doc_no, 'score': score})

    df = pd.DataFrame(rows)

    # Sort and rank
    df['rank'] = df.groupby('query_id')['score'].rank(ascending=False)

    df.sort_values(by=['query_id', 'rank'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(f'./BERT_data/rank_df_{cp}.csv', index=False)
    return df

def create_ranking_file(df):
    df['Q0'],df['system'] = 'Q0', 'BERT'
    df = df[['query_id', 'Q0', 'doc_no', 'rank', 'score', 'system']]
    df['rank'] = df['rank'].astype(int)
    output_file_path = f"./BERT_data/BERT{cp}"
    df.to_csv(output_file_path, sep=' ', index=False, header=False)


def get_bert_scores(queries_to_docnos, docs_dict, queries_dict):
    """
    Get BERT scores for a given queries and documents.
    Important: there can't be a document that belongs to more than one query!
    :param queries_to_docnos: dict of queries ids to docnos
    :param docs_dict: dict of docnos to documents text
    :param queries_dict: dict of queries ids to queries text
    :return: dict of docnos to BERT scores
    """
    scores_dict = {qid: {} for qid in queries_dict.keys()}

    for qid in tqdm(queries_to_docnos, desc="BERT", total=len(queries_to_docnos)):
        for docno in queries_to_docnos[qid]:
            query_text = queries_dict[qid]
            doc_text = docs_dict[docno]
            ret = tokenizer.encode_plus(query_text,
                                        doc_text,
                                        max_length=512,
                                        truncation=True,
                                        return_token_type_ids=True,
                                        return_tensors='pt')

            with torch.cuda.amp.autocast(enabled=False):
                input_ids = ret['input_ids'].to(device)
                tt_ids = ret['token_type_ids'].to(device)
                output, = model(input_ids, token_type_ids=tt_ids, return_dict=False)
                if output.size(1) > 1:
                    score = torch.nn.functional.log_softmax(output, 1)[0, -1].item()
                else:
                    score = output.item()

            scores_dict[qid][docno] = score

    return scores_dict


if __name__ == '__main__':
    queries_to_docnos = parse_working_set(working_set_file_path)
    docs_dict = parse_bot_followup(bot_followup_file_path)
    queries_dict = parse_queries(queries_file_path)
    scores_dict = get_bert_scores(queries_to_docnos, docs_dict, queries_dict)
    rank_df = create_rank_df(scores_dict)
    create_ranking_file(rank_df)

    x = 1
