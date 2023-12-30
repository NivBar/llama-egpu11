import fire
import pandas as pd
from llama import Llama
from tqdm import tqdm
import post_process
import warnings
# import torch.nn as nn
warnings.filterwarnings("ignore")
import os
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'


def get_chunks(input_list, k):
    for i in range(0, len(input_list), k):
        yield input_list[i:i + k]


def main(ckpt_dir: str,
         tokenizer_path: str,
         data_path: str,
         temperature: float = 0.6,
         top_p: float = 0.9,
         max_seq_len: int = 128,
         max_gen_len: int = 1000,
         max_batch_size: int = 1):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        prompts_path (str): path for prompts (line break separated)
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """
    orig = pd.read_csv(data_path)
    df = orig[orig.text.isna()].copy()
    prompts = df["prompt"].tolist()
    indexes = df.index.tolist()
    prompt_batches = [prompts[i:i + max_batch_size] for i in range(0, len(prompts), max_batch_size)]
    index_batches = [indexes[i:i + max_batch_size] for i in range(0, len(indexes), max_batch_size)]

    max_seq_len = max(len(p) for p in prompts)
    print("\n\n### max_seq_len: ", max_seq_len, ",max_batch_size: ", max_batch_size, ",row_num: ", len(prompts),
          " ###\n\n")

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size
    )

    print("starting to generate...")
    for batch, index_list in tqdm(zip(prompt_batches, index_batches), total=len(prompt_batches)):
        results = generator.text_completion(
            batch,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p, )

        for idx, res in zip(index_list, results):
            print(f"{idx + 1}/{len(prompts)}")

            try:
                new_text = post_process.count_words_complete_sentences(
                    res["generation"].split("[/INST]")[1]).strip()
                new_text = new_text.strip().replace("\n", " ")
            except:
                new_text = None
            if new_text is not None:
                print(f"wc: {len(new_text.split())}, text> \n{new_text}")
                orig.at[idx, "text"] = new_text
            else:
                print(f"text too short! rerun with higher max_gen_len")
            print("\n==================================\n")

        orig.to_csv(data_path, index=False)

    df.to_csv(data_path, index=False)
    print("DONE!")


if __name__ == "__main__":
    # fire.Fire(main)
    main(ckpt_dir="llama-2-7b/",
         tokenizer_path="tokenizer.model",
         data_path="bot_followup_llama1.csv")
