#!/usr/bin/env python3

import os
import sys
import csv
import json
import re
import subprocess

def main():
    max_round = 6
    csv_file = 'tommy_data.csv'
    base_path = '/lv_local/home/niv.b/llama/'
    jsonl_filename = f'tommy_data_{max_round}_max.jsonl'

    # Define folder names
    folder1 = os.path.join(base_path, f'pyserini_output_tommy_{max_round}_max')
    folder2 = os.path.join(base_path, f'pyserini_output_tommy_{max_round}_max_sparse')

    # Create directories if they don't exist
    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)

    # Process CSV and write JSONL files
    data = process_csv(csv_file, max_round)
    write_jsonl_file(data, folder1, jsonl_filename)
    write_jsonl_file(data, folder2, jsonl_filename)

    # Run Pyserini encoding
    run_pyserini_encode(folder1, jsonl_filename)

    # Run Pyserini indexing
    run_pyserini_index(folder2)

def process_csv(csv_file, max_round):
    """Process the CSV file and return filtered data."""
    data = []
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                round_no = int(row['round_no'])
                if round_no <= max_round:
                    doc_id = row['docno']
                    contents = row['current_document']
                    # Remove \n and \r from contents
                    contents = re.sub(r'[\r\n\\]+', ' ', contents)
                    contents = re.sub(r'\s+', ' ', contents).strip()
                    item = {
                        "id": doc_id,
                        "text": contents
                    }
                    data.append(item)
            except ValueError:
                continue
    return data

def write_jsonl_file(data, folder, jsonl_filename):
    """Write the JSONL file to the specified folder."""
    jsonl_path = os.path.join(folder, jsonl_filename)
    with open(jsonl_path, 'w', encoding='utf-8') as outfile:
        for item in data:
            json.dump(item, outfile)
            outfile.write('\n')

def run_pyserini_encode(folder, jsonl_filename):
    """Run the Pyserini encode command."""
    corpus_path = os.path.join(folder, jsonl_filename)
    command = [
        'python', '-m', 'pyserini.encode',
        'input', '--corpus', corpus_path,
                 '--fields', 'text',
        'output', '--embeddings', folder, '--to-faiss',
        'encoder', '--encoder', 'intfloat/e5-base',
                  '--fields', 'text',
                  '--batch', '32',
                  '--fp16'
    ]
    subprocess.run(command)

def run_pyserini_index(folder):
    """Run the Pyserini index command."""
    command = [
        'python', '-m', 'pyserini.index.lucene',
        '--collection', 'JsonCollection',
        '--input', folder,
        '--index', folder
    ]
    subprocess.run(command)

if __name__ == "__main__":
    main()
