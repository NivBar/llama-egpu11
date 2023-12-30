from pyserini.search import FaissSearcher
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
import torch
import pandas as pd
import json
from tqdm import tqdm
import os
import numpy as np

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


def get_prob_mean(text1, text2):
    return (get_prob_one_sided(text1, text2) + get_prob_one_sided(text2, text1)) / 2


def get_prob_one_sided(premise, hypothesis):
    input_ids = tokenizer(
        f'premise: {premise} hypothesis: {hypothesis}',
        return_tensors='pt',
        truncation=True,
        max_length=512).input_ids
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    logits = outputs.logits
    probs = torch.softmax(logits[0], dim=-1)
    one_token_id = tokenizer('1').input_ids[0]
    entailment_prob = probs[0, one_token_id].item()
    return entailment_prob


def init_model():
    model_path = 'google/t5_11b_trueteacher_and_anli'
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    model.eval()
    return tokenizer, model


def truncate_to_max_tokens(text, tokenizer, max_tokens=512):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)


if __name__ == '__main__':
    cp = 'llama13btest'

    if os.path.exists(f'/lv_local/home/niv.b/llama/TT_input/feature_data_{cp}_new_F.csv'):
        df = pd.read_csv(f'/lv_local/home/niv.b/llama/TT_input/feature_data_{cp}_new_F.csv').astype(str)
    else:
        df = pd.read_csv(f'/lv_local/home/niv.b/llama/TT_input/feature_data_{cp}_new.csv').astype(str)
        df = df[df.creator != 'creator']
        df['faithfulness'] = np.nan

    nan_indices = df[df.faithfulness == 'nan'].index.tolist()
    print("remaining: ",len(nan_indices))

    text_df = pd.read_csv(f'/lv_local/home/niv.b/llama/TT_input/bot_followup_{cp}.csv').astype(str)
    text_df = text_df.merge(df[['query_id', 'creator', 'username', 'docno']], on=['query_id', 'creator', 'username'],
                            how='left')
    greg_df = pd.read_csv(f'/lv_local/home/niv.b/llama/greg_data.csv').astype(str)
    greg_df['new_docno'] = greg_df.apply(lambda row: row.docno.split('ROUND-')[1].replace("00", "0") + "-creator",
                                         axis=1)

    tokenizer, model = init_model()

    counter = 0
    error_counter = 0

    for idx, row in tqdm(df.iterrows()):
        if idx not in nan_indices:
            continue
        try:
            k = 10
            text = text_df[text_df.docno == row.docno].text.values[0]
            text = truncate_to_max_tokens(text, tokenizer, max_tokens=512)
            try:
                prev_text = greg_df[greg_df.new_docno == row.previous_docno_str].current_document.values[0]
            except:
                prev_text = greg_df[greg_df.new_docno == row.previous_docno_str.replace("-"+row.previous_docno_str.split("-")[2]+"-","-0" + row.previous_docno_str.split("-")[2] + "-")].current_document.values[0]

            top_docs = retrieve_top_docs(text, k=k)
            top_docs = [greg_df[greg_df.docno == docno].current_document.values[0] for docno in top_docs]

            with torch.no_grad():
                faithfulness = get_prob_mean(text, prev_text)

                EF = sum(get_prob_mean(text, doc) for doc in top_docs) / k

                print(
                    f'SUCCESS! docno: {row.docno}, EF: {EF}, faithfulness: {faithfulness}')

                df.loc[idx, f'EF@{k}'] = EF
                df.loc[idx, 'faithfulness'] = faithfulness
                counter += 1
                if counter % 5 == 0:
                    print(f"finished: {counter}/{len(nan_indices)}, errors: {error_counter}")
                    df.to_csv(f'/lv_local/home/niv.b/llama/TT_input/feature_data_{cp}_new_F.csv', index=False)
        except Exception as e:
            print(f"ERROR!: {e}, docno: {row.docno}")
            error_counter += 1
            continue
