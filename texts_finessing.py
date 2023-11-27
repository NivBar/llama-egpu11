import pandas as pd
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from pprint import pprint


def process_string(text, lower=True):
    if lower:
        res = re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\!\?\-\'\"]', '', text.lower()).strip()
    else:
        res = re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\!\?\-\'\"]', '', text).strip()
    return res

for i in range(10):
    print("Iteration: ", i)
    words_colon = []
    words_hyphen = []
    counter = 0
    nan_index = []

    if i == 0:
        start = True
    else:
        start = False

    if start:
        dfs = []
        for file in os.listdir():
            if file.endswith('.csv') and file not in ['bot_followup_llama.csv', 'original_df.csv']:
                df = pd.read_csv(file)
                dfs.append(df[['round_no', 'query_id', 'creator', 'username', 'text', 'prompt']])
        orig = pd.concat(dfs, ignore_index=True)
        orig = orig.dropna(subset=['text']).sort_values(by=['round_no', 'query_id', 'creator'])
    else:
        orig = pd.read_csv('bot_followup_llama.csv')

    df = orig.copy()

    for idx, row in tqdm(df.iterrows()):
        try:
            if ":" in row['text']:
                match_colon = row['text'].split(':', 1)[0]
                words_colon.append(process_string(match_colon))

                match_hyphen = row['text'].split('-', 1)[0]
                words_hyphen.append(process_string(match_hyphen))

        except(TypeError):
            # print("ERROR in ",idx, " :TypeError")
            counter += 1
            nan_index.append(idx)
            pass

    # empty_df = df.iloc[nan_index]
    # empty_df.to_csv('empty_df.csv', index=False)

    word_counts_colon = Counter(words_colon)
    most_common_pairs_colon = word_counts_colon.most_common(200)
    most_common_words_colon = [w for w, v in most_common_pairs_colon if
                               (v >= 30) or ("title" in w) or ("enhanc" in w) or ("query" in w) or (
                                           "document" in w) or (
                                       "paragraph" in w) or ("markdown" in w)]
    print("Most common words with colon: ", len(most_common_words_colon), most_common_words_colon)

    word_counts_hyphen = Counter(words_hyphen)
    most_common_pairs_hyphen = word_counts_hyphen.most_common(200)
    most_common_words_hyphen = [w for w, v in most_common_pairs_hyphen if
                                v >= 30 or ("title" in w) or ("enhanc" in w) or ("query" in w) or ("document" in w) or (
                                        "paragraph" in w) or ("markdown" in w)]
    print("Most common words with hyphen: ", len(most_common_words_hyphen), most_common_words_hyphen)

    if len(most_common_words_colon) == 0 and len(most_common_words_hyphen) == 0:
        print("No common words found")
        exit()

    for idx, row in tqdm(df.iterrows()):
        try:
            match_colon = row['text'].split(':', 1)
            match_hyphen = row['text'].split('-', 1)

            if process_string(match_colon[0]) in most_common_words_colon and process_string(
                    match_hyphen[0]) in most_common_words_hyphen:
                df.at[idx, 'text'] = process_string(match_colon[1], lower=False) if len(match_colon[1]) > len(
                    match_hyphen[1]) else process_string(match_hyphen[1], lower=False)
            elif process_string(match_colon[0]) in most_common_words_colon:
                df.at[idx, 'text'] = process_string(match_colon[1], lower=False)
            elif process_string(match_hyphen[0]) in most_common_words_hyphen:
                df.at[idx, 'text'] = process_string(match_hyphen[1], lower=False)
            else:
                df.at[idx, 'text'] = process_string(row['text'], lower=False)
        except:
            pass

    df.to_csv('bot_followup_llama.csv', index=False)

# pprint(most_common_words)
# for lst in [most_common_pairs_colon, most_common_pairs_hyphen]:
#     words, counts = zip(*lst)
#     plt.figure(figsize=(15, 10))
#     plt.bar(words, counts)
#     plt.xlabel('Words')
#     plt.ylabel('Counts')
#     plt.title('100 Most Common Words')
#     plt.xticks(rotation=90)  # Rotate labels to avoid overlap
#     plt.show()
