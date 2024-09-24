import sys
from pyserini.search import FaissSearcher
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
import torch
import pandas as pd
import json
from tqdm import tqdm
import os
import numpy as np
import re
from transformers import DPRQuestionEncoderTokenizer
import warnings
from transformers import AutoTokenizer
from pyserini.search.lucene import LuceneSearcher
from TrueTeacher_utils import *
import socket

if __name__ == '__main__':

    # cp = 'online'
    # online = True

    cp = 'tommy4@F200'
    online = False
    EF_debug = True
    print(f"starting {cp} analysis...")

    # df = pd.read_csv(f'/lv_local/home/niv.b/llama/TT_input/feature_data_{cp}_new.csv').astype(str)
    # df_F = pd.read_csv(f'/lv_local/home/niv.b/llama/TT_input/feature_data_{cp}_new_F.csv').astype(str)
    # asrc_scores = pd.read_csv(f'/lv_local/home/niv.b/llama/TT_input/asrc_f_scores.csv').astype(str)
    #
    # df = df.merge(pd.concat([asrc_scores,df_F])[['docno','faithfulness','EF@10']].dropna(), on=['docno'], how='left', suffixes=('', '_y')).drop_duplicates()
    #
    # df.to_csv(f'/lv_local/home/niv.b/llama/TT_input/feature_data_{cp}_new_F.csv', index=False)

    # asrc_scores = pd.read_csv(f'/lv_local/home/niv.b/llama/TT_input/feature_data_asrc_new_F.csv').astype(str) # TOMMY

    if os.path.exists(f'/lv_local/home/niv.b/llama/TT_input/feature_data_{cp}_new_F.csv'):
        df = pd.read_csv(f'/lv_local/home/niv.b/llama/TT_input/feature_data_{cp}_new_F.csv').astype(str)

    else:
        if os.path.exists(f'/lv_local/home/niv.b/llama/TT_input/feature_data_{cp}_new.csv'):
            df = pd.read_csv(f'/lv_local/home/niv.b/llama/TT_input/feature_data_{cp}_new.csv').astype(str)
            # df = df.merge(asrc_scores, on=['docno'], how='left', #TOMMY
            #               suffixes=('', '_y')).astype(str)
            # df.username = cp # TOMMY

        else:
            df = create_feature_df(cp, online)

        # df = df[df.creator != 'creator'] # bot calculations!

        # df = df[df.creator == 'creator'].drop_duplicates("docno
        # ") # student calculations!

        # df['faithfulness'] = "nan"

    # eval_measures = ['CF_BDM', 'EF@10', 'CF@1', 'CF@2', 'CF@1_values', 'CF@1_ref', 'NCF@1']
    eval_measures = ['NEF@10', 'NEF@10_max', 'EF@10', 'EF@10_max', 'CF@1', 'CF@1_ref', 'NCF@1',
                     'EF@10_dense', 'EF@10_sparse',
                     'EF@10_max_dense', 'EF@10_max_sparse',
                     'NEF@10_dense', 'NEF@10_sparse',
                     'NEF@10_max_dense', 'NEF@10_max_sparse',
                     'EF@5_dense', 'EF@5_sparse',
                     'EF@5_max_dense', 'EF@5_max_sparse',
                     'NEF@5_dense', 'NEF@5_sparse',
                     'NEF@5_max_dense', 'NEF@5_max_sparse', 'ref_pos_sparse', 'ref_pos_dense']

    for measure in eval_measures:
        if measure not in df.columns:
            df[measure] = "nan"

    # df["NCF@1"] = df.apply(
    #     lambda row: float(row["CF@1"]) / float(row["CF@1_ref"]) if pd.notnull(float(row["CF@1"])) and pd.notnull(
    #         float(row["CF@1_ref"])) else 'nan',
    #     axis=1)

    if cp == 'asrc':
        # round_no, query_id, creator, username
        df[['round_no', 'query_id', 'username', 'creator']] = df.docno.apply(lambda x: pd.Series(x.split('-')))
        df[['round_no', 'query_id', 'username']] = df[['round_no', 'query_id', 'username']].astype(int).astype(str)

    try:
        y_cols = [col for col in df.columns if '_y' in col]
        df = df.drop(y_cols, axis=1)
    except:
        pass
    # nan_indices = df[df.faithfulness == 'nan'].index.tolist()

    # if cp != 'asrc': #TOMMY_DATA
    #     df = df[df.creator != 'creator']
    nan_indices = df.index[df[eval_measures].apply(lambda row: any(row == 'nan'), axis=1)].tolist()

    if len(nan_indices) == 0:
        print("all done!")
        df.to_csv(f'/lv_local/home/niv.b/llama/TT_input/feature_data_{cp}_new_F.csv', index=False)
        columns_to_convert = [col for col in eval_measures if col != 'CF@1_values']
        df[columns_to_convert] = df[columns_to_convert].astype(float)
        df[eval_measures] = df[eval_measures].replace('nan', np.nan)
        # mean_df = df.groupby(['temp', 'username'])[cols_to_convert].mean().reset_index()
        mean_df = df.groupby(['username'])[columns_to_convert].mean().reset_index()
        print(mean_df)
        exit()

    print("remaining: ", len(nan_indices))
    try:
        text_df = pd.read_csv(f'/lv_local/home/niv.b/llama/TT_input/bot_followup_{cp}.csv').astype(str)
    except:
        text_df = pd.read_csv(f'/lv_local/home/niv.b/llama/TT_input/feature_data_{cp}_new.csv').astype(str)
    text_df = text_df.astype(str).merge(df[['query_id', 'creator', 'username', 'docno']].astype(str),
                                        on=['query_id', 'creator', 'username'],
                                        how='left', suffixes=('', '_y'))
    if text_df.docno.isna().all():
        text_df.drop('docno', axis=1, inplace=True)
        if "temp" in text_df.columns:
            text_df['username'] = text_df['username'].astype(str) + "@" + text_df['temp'].apply(
                lambda x: str(int(float(x) * 100)))  # ***
        text_df = text_df.astype(str).merge(df[['query_id', 'creator', 'username', 'docno']].astype(str),
                                            on=['query_id', 'creator', 'username'],
                                            how='left', suffixes=('', '_y'))
        x = 1

    # # long texts handling
    # long_df = text_df[text_df['text'].apply(lambda x: len(x.split()) > 150)]
    # long_df['length'] = long_df['text'].apply(lambda x: len(x.split()))
    # long_df['text'] = long_df['text'].apply(trim_complete_sentences)
    # long_df['length_post_edit'] = long_df['text'].apply(lambda x: len(x.split()))
    #
    # text_df['text'] = text_df['text'].apply(lambda x: trim_complete_sentences(x) if len(x.split()) > 150 else x)
    #
    # assert text_df['text'].apply(lambda x: len(x.split())).max() <= 150
    #
    # # TODO: remove long fix-up!!!
    # df.loc[df.docno.isin(long_df.docno.tolist()), eval_measures] = 'nan'

    if not online:
        greg_df = pd.read_csv(f'/lv_local/home/niv.b/llama/tommy_data.csv').astype(str)
    else:
        greg_df = pd.read_csv(f'/lv_local/home/niv.b/llama/full_online_data.csv').astype(str)
    greg_df['new_docno'] = greg_df.apply(lambda row: row.docno.split('ROUND-')[1].replace("00", "0") + "-creator",
                                         axis=1)

    # # round_no, query_id, creator, username, text, prompt
    # greg_df = greg_df[['round_no', 'query_id', 'username', 'current_document','new_docno']].rename({'new_docno':'docno', 'current_document':'text'}, axis=1)
    # greg_df['creator'] = 'creator'
    # greg_df.to_csv(f'/lv_local/home/niv.b/llama/TT_input/bot_followup_{cp}.csv', index=False)

    if cp == 'asrc':
        cf_ref_dict = create_CF_ref_dict(df)
        df['CF@1_ref'] = df.apply(lambda row: cf_ref_dict.get((row['query_id'], row['creator']),
                                                              cf_ref_dict.get((row['query_id'], row['username']),
                                                                              {})).get(
            'CF@1_ref', 'nan'), axis=1)
    # TODO: pay attention to model in util file
    # tokenizer, model = init_model()

    # winner_df = greg_df[
    #     (greg_df.position.astype(float).astype(int) == 1) & (greg_df.round_no.astype(float).astype(int) == 4)]
    # winner_df.loc[:, 'current_document'] = winner_df['current_document'].apply(
    #     lambda x: truncate_to_max_tokens(x, tokenizer, max_tokens=512))
    #
    # winner_dict = winner_df[['query_id', 'current_document']].set_index('query_id').to_dict('index')

    counter = 0
    error_counter = 0
    if socket.gethostname() != remote_host:
        cache = read_pickle_from_remote(remote_host, remote_pickle_path, username, password)
    else:
        cache = load_cache()
    cache_size = get_cache_size(cache)
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row.round_no == '1':
            continue
        if idx not in nan_indices:
            continue

        if row.previous_docno_str not in data_dict.keys():  # EF_DEBUG
            assert row.previous_docno_str in names_nto
        try:
            k = 10
            try:
                text = text_df[text_df.docno == row.docno].text.values[0]  # bot calculations!
            except:
                text = greg_df[greg_df.new_docno == row.docno.replace("-" + row.docno.split("-")[2] + "-",
                                                                      "-0" + row.docno.split("-")[
                                                                          2] + "-")].current_document.values[0]

            sentences = [sentence.strip() for sentence in re.split(r'[.!?]', text) if sentence]
            sentences_duo = [sentences[i] + '. ' + sentences[i + 1] for i in range(len(sentences) - 1)]

            # try: # student calculations!
            #     text = greg_df[greg_df.new_docno == row.docno].current_document.values[0]
            # except:
            #     text = greg_df[greg_df.new_docno == row.docno.replace("-"+row.docno.split("-")[2]+"-","-0" + row.docno.split("-")[2] + "-")].current_document.values[0]

            text = truncate_to_max_tokens(text, max_tokens=512)
            try:
                if 'creator' in row.docno or row.creator == 'creator':  # TOMMY DATA
                    try:  # TOMMY DATA
                        # prev_text = greg_df[greg_df.new_docno == row.docno.replace(f"0{int(row.round_no)}", f"0{int(row.round_no) - 1}")].current_document.values[0] #TOMMY DATA
                        prev_text = greg_df[greg_df.docno == row.docno.replace(f"0{int(row.round_no)}",
                                                                               f"0{int(row.round_no) - 1}")].current_document.values[
                            0]
                        x = 1
                    except:
                        prev_text = greg_df[greg_df.new_docno == row.docno.replace(f"0{int(row.round_no)}",
                                                                                   f"0{int(row.round_no) - 1}").replace(
                            "-" + row.docno.split("-")[2] + "-",
                            "-0" + row.docno.split("-")[2] + "-")].current_document.values[0]
                else:
                    prev_text = greg_df[greg_df.new_docno == row.previous_docno_str].current_document.values[0]
            except:
                if 'previous_docno_str' in df.columns and row.previous_docno_str != 'nan':
                    try:
                        if online:
                            prev_text = greg_df[greg_df.docno == row.previous_docno_str].current_document.values[0]
                        else:
                            prev_text = greg_df[greg_df.new_docno == row.previous_docno_str].current_document.values[0]
                    except:
                        prev_text = greg_df[greg_df.new_docno == row.previous_docno_str.replace(
                            "-" + row.previous_docno_str.split("-")[2] + "-",
                            "-0" + row.previous_docno_str.split("-")[2] + "-")].current_document.values[0]

            try:  # EF_DEBUG
                prev_text = data_dict[row.previous_docno_str]
            except:
                prev_text = data_dict[names_nto[row.previous_docno_str]]
            prev_text = truncate_to_max_tokens(prev_text, max_tokens=512)

            # winner_text = winner_dict[row.query_id]['current_document']
            # winner_text = truncate_to_max_tokens(winner_text, tokenizer, max_tokens=512)

            with torch.no_grad():
                # if row['CF_BDM'] == 'nan':
                #     CF_BDM = get_prob_mean(text, prev_text)
                #     df.loc[idx, 'CF_BDM'] = CF_BDM

                # if row[f'EF@{k}'] == 'nan' or row[f'EF@{k}_max'] == 'nan':
                #     top_docs_docnos, hits = retrieve_top_docs(text, k=k, round_no=row.round_no)
                #     top_docs = [greg_df[greg_df.docno == docno].current_document.values[0] for docno in top_docs_docnos]
                #     top_docs = [truncate_to_max_tokens(doc, tokenizer, max_tokens=512) for doc in top_docs]
                #
                #     vals = [get_prob_mean(text, doc) for doc in top_docs]
                #
                #     EF = sum(vals) / k
                #     EF_max = max(vals)
                #
                #     df.loc[idx, f'EF@{k}'] = EF
                #     df.loc[idx, f'EF@{k}_max'] = EF_max

                if row['NEF@10_dense'] == 'nan' or row['NEF@10_max_dense'] == 'nan' or row['EF@10_dense'] == 'nan' or \
                        row['EF@10_max_dense'] == 'nan' or row['NEF@5_dense'] == 'nan' or row[
                    'NEF@5_max_dense'] == 'nan' or row['EF@5_dense'] == 'nan' or row['EF@5_max_dense'] == 'nan' or \
                        row['NEF@10_sparse'] == 'nan' or row['NEF@10_max_sparse'] == 'nan' or row[
                    'EF@10_sparse'] == 'nan' or row['EF@10_max_sparse'] == 'nan' or row['NEF@5_sparse'] == 'nan' or row[
                    'NEF@5_max_sparse'] == 'nan' or row['EF@5_sparse'] == 'nan' or row[
                    'EF@5_max_sparse'] == 'nan' or EF_debug:
                    # top_docs_docnos, hits = retrieve_top_docs(text, k=k, round_no=row.round_no)
                    # top_docs = [greg_df[greg_df.docno == docno].current_document.values[0] for docno in top_docs_docnos]
                    # top_docs = [truncate_to_max_tokens(doc, tokenizer, max_tokens=512) for doc in top_docs]
                    #
                    # cf_vals, cf_ref_vals, norm_vals = [], [], []
                    #
                    # for doc in top_docs:
                    #     vals = [get_prob_one_sided(doc, sentence) for sentence in sentences]
                    #     bool_vals = [v >= 0.5 for v in vals]
                    #     res = sum(bool_vals) / len(bool_vals)
                    #
                    #     ref_sentences = [sentence.strip() for sentence in re.split(r'[.!?]', doc) if sentence]
                    #     ref_vals = [get_prob_one_sided(doc, sentence) for sentence in ref_sentences]
                    #     ref_bool_vals = [v >= 0.5 for v in ref_vals]
                    #     ref_res = sum(ref_bool_vals) / len(ref_bool_vals)
                    #
                    #     norm_res = min(res / ref_res if ref_res != 0 else 1.0, 1.0)
                    #
                    #     cf_vals.append(res)
                    #     cf_ref_vals.append(ref_res)
                    #     norm_vals.append(norm_res)
                    #
                    # df.loc[idx, f'NEF@{k}'] = sum(norm_vals) / len(norm_vals)
                    # df.loc[idx, f'NEF@{k}_max'] = max(norm_vals)
                    # df.loc[idx, f'EF@{k}'] = sum(cf_vals) / len(cf_vals)
                    # df.loc[idx, f'EF@{k}_max'] = max(cf_vals)
                    # x = 1

                    # Retrieve top documents for both dense (FAISS) and sparse (TFIDF) searchers

                    top_docs_dense_docnos, top_docs_sparse_docnos, _, _ = retrieve_top_docs(text, k=k,
                                                                                            round_no=row.round_no,
                                                                                            online=online)

                    # Process documents and calculate metrics for both dense and sparse retrieval
                    metrics_dense = process_top_docs(top_docs_dense_docnos, greg_df, sentences, k, cache)
                    metrics_sparse = process_top_docs(top_docs_sparse_docnos, greg_df, sentences, k, cache)

                    # Store results for both dense and sparse methods
                    store_results(df, idx, metrics_dense, "dense", k, k // 2)
                    store_results(df, idx, metrics_sparse, "sparse", k, k // 2)

                # if row['TF_BDM'] == 'nan':
                #     TF_BDM = get_prob_mean(text, winner_text)
                #     df.loc[idx, 'TF_BDM'] = TF_BDM

                if row['ref_pos_sparse'] == 'nan' or row['ref_pos_dense'] == 'nan' or EF_debug:
                    top_docs_dense_docnos, top_docs_sparse_docnos, _, _ = retrieve_top_docs(text, k=k,
                                                                                            round_no=row.round_no,
                                                                                            online=online)
                    df.loc[idx, 'ref_pos_sparse'] = (
                        top_docs_sparse_docnos.index(row.previous_docno_str) + 1
                        if row.previous_docno_str in top_docs_sparse_docnos
                        else (
                            top_docs_sparse_docnos.index(names_nto[row.previous_docno_str]) + 1
                            if row.previous_docno_str in names_nto and names_nto[
                                row.previous_docno_str] in top_docs_sparse_docnos
                            else 999
                        )
                    )

                    df.loc[idx, 'ref_pos_dense'] = (
                        top_docs_dense_docnos.index(row.previous_docno_str) + 1
                        if row.previous_docno_str in top_docs_dense_docnos
                        else (
                            top_docs_dense_docnos.index(names_nto[row.previous_docno_str]) + 1
                            if row.previous_docno_str in names_nto and names_nto[
                                row.previous_docno_str] in top_docs_dense_docnos
                            else 999
                        )
                    )
                    x = 1

                if row['CF@1'] == 'nan' or row['NCF@1'] == 'nan' or EF_debug:
                    cf_vals, cf_ref_vals, norm_vals = calculate_metrics([prev_text], sentences, cache)
                    df.loc[idx, 'CF@1'] = cf_vals[0]
                    df.loc[idx, 'CF@1_ref'] = cf_ref_vals[0]
                    df.loc[idx, 'NCF@1'] = norm_vals[0]

                    x = 1

                if cache_size != get_cache_size(cache):
                    if socket.gethostname() == remote_host:
                        save_cache(cache)
                    else:
                        write_pickle_to_remote(data, remote_host, remote_pickle_path, username, password)

                    cache_size = get_cache_size(cache)

                # if row['CF@1'] == 'nan' or row['CF@1_values'] == 'nan' or EF_debug:
                #     vals = [get_prob_one_sided(prev_text, sentence) for sentence in sentences]
                #     bool_vals = [v >= 0.5 for v in vals]
                #     res = sum(bool_vals) / len(bool_vals)
                #     # print(vals,'\n',bool_vals,'\n', res)
                #     df.loc[idx, 'CF@1'] = res
                #     df.loc[idx, 'CF@1_values'] = str(vals)
                #
                # if row['NCF@1'] == 'nan' or EF_debug:
                #     if row['CF@1_ref'] == 'nan':
                #         ref_sentences = [sentence.strip() for sentence in re.split(r'[.!?]', prev_text) if sentence]
                #         vals = [get_prob_one_sided(prev_text, sentence) for sentence in ref_sentences]
                #         bool_vals = [v >= 0.5 for v in vals]
                #         res = sum(bool_vals) / len(bool_vals)
                #         # print(vals,'\n',bool_vals,'\n', res)
                #         df.loc[idx, 'CF@1_ref'] = res
                #         if cp != 'asrc':
                #             cf_ref_dict = create_CF_ref_dict(df)
                #             df['CF@1_ref'] = df.apply(lambda row: cf_ref_dict.get((row['query_id'], row['creator']),
                #                                                                   cf_ref_dict.get(
                #                                                                       (
                #                                                                           row['query_id'],
                #                                                                           row['username']),
                #                                                                       {})).get(
                #                 'CF@1_ref', 'nan'), axis=1)
                #
                #     df.loc[idx, 'NCF@1'] = min(float(df.loc[idx, 'CF@1']) / float(df.loc[idx, 'CF@1_ref']), 1.0)
                #     x = 1

                for suffix in ['dense', 'sparse']:
                    if df.loc[idx, f'ref_pos_{suffix}'] != 999:
                        nef_value = df.loc[idx, f'NEF@{k}_max_{suffix}']
                        ncf_value = df.loc[idx, 'NCF@1']
                        ef_value = df.loc[idx, f'EF@{k}_max_{suffix}']
                        cf_value = df.loc[idx, 'CF@1']

                        if nef_value < ncf_value:
                            print(
                                f"NEF@{k}_max_{suffix} (value: {nef_value}) is less than NCF@1 (value: {ncf_value}) for index {idx}")

                        if ef_value < cf_value:
                            print(
                                f"EF@{k}_max_{suffix} (value: {ef_value}) is less than CF@1 (value: {cf_value}) for index {idx}")

                        x = 1

                # if row['CF@2'] == 'nan':
                #     if len(sentences_duo) == 0:
                #         df.loc[idx, 'CF@2'] = 0
                #         print('CF@2: no sentences duo! - ', text)
                #     else:
                #         res = sum([get_prob_one_sided(prev_text, sentence) > 0.5 for sentence in sentences_duo])
                #         df.loc[idx, 'CF@2'] = res / len(sentences_duo)

                # if row['TF@1'] == 'nan':
                #     res = sum([get_prob_one_sided(winner_text, sentence) > 0.5 for sentence in sentences])
                #     df.loc[idx, 'TF@1'] = res / len(sentences)
                #
                # if row['TF@2'] == 'nan':
                #     if len(sentences_duo) == 0:
                #         df.loc[idx, 'TF@2'] = 0
                #         print('TF@2: no sentences duo! - ', text)
                #     else:
                #         res = sum([get_prob_one_sided(winner_text, sentence) > 0.5 for sentence in sentences_duo])
                #         df.loc[idx, 'TF@2'] = res / len(sentences_duo)

                # print(f'SUCCESS! docno: {row.docno}') #EF_DEBUG

                counter += 1
                if counter % 10 == 0 or counter == len(nan_indices) - error_counter:
                    # print(f"finished: {counter}/{len(nan_indices)}, errors: {error_counter}") #EF_DEBUG
                    df.to_csv(f'/lv_local/home/niv.b/llama/TT_input/feature_data_{cp}_new_F.csv', index=False)
                    df[eval_measures] = df[eval_measures].replace('nan', np.nan)
                    columns_to_convert = [col for col in eval_measures if col != 'CF@1_values']
                    df[columns_to_convert] = df[columns_to_convert].astype(float)
                    # mean_df = df.groupby(['temp', 'username'])[cols_to_convert].mean().reset_index()
                    mean_df = df.groupby(['username'])[columns_to_convert].mean().reset_index()
                    # print(mean_df) #EF_DEBUG


        except Exception as e:
            # print(f"ERROR!: {e}, docno: {row.docno}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, exc_tb.tb_lineno, e)
            error_counter += 1
            continue
    x = 1
