import re
import numpy as np
import regex



def filter_utf8(text):
    # Regular expression to match only UTF-8 alphabet, numbers, and punctuation
    pattern = regex.compile(r'[\p{L}\p{N}\p{P}\p{Z}]+', regex.UNICODE)
    filtered_text = ''.join(pattern.findall(text))
    return filtered_text

def remove_sentences_second(sentences):
    # Calculate the initial total number of words
    total_words = sum(len(sentence.split()) for sentence in sentences)

    # Check if total_words is already within the desired range or below 140
    if total_words <= 150:
        return sentences

    # Find and remove a sentence if total_words can be adjusted within the desired range
    while total_words > 150:
        for sentence in sorted(sentences, key=lambda sentence: len(sentence.split())):
            # Calculate the updated total number of words without the current sentence
            updated_total_words = total_words - len(sentence.split())
            if updated_total_words <= 150:
                total_words = updated_total_words
                sentences.remove(sentence)
                return sentences

        # Remove the shortest sentence if no sentence can be removed to meet the desired range or below 140
        if total_words > 150:
            shortest_sentence = min(sentences, key=lambda sentence: len(sentence.split()))
            sentences.remove(shortest_sentence)
            total_words = total_words - len(shortest_sentence.split())
    return sentences

def count_words_complete_sentences(text):
    # Split the text into sentences using a regex pattern
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)

    # Check if the last sentence is incomplete and remove it if necessary
    if sentences and sentences[-1].strip() and sentences[-1].strip()[-1] not in ['.', '!', '?']:
        if sum(len(sentence.split()) for sentence in sentences[:-1]) < 140:
            sentences[-1] = sentences[-1] + '.'
        else:
            sentences.pop()

    sentences_second = sentences.copy()

    # Check if truncation is necessary
    if sentences:
        word_count = sum(len(sentence.split()) for sentence in sentences)
        truncated_text = " ".join(sentences)

        while word_count > 150:
            if len(sentences) < 2:
                break
            sentences.pop()
            truncated_text = ' '.join(sentences)
            word_count = sum(len(sentence.split()) for sentence in sentences)

        if word_count < 140:
            # second try (less preferred)
            word_count_second = sum(len(sentence.split()) for sentence in sentences_second)
            sentences_second = remove_sentences_second(sentences_second)
            truncated_text_second = ' '.join(sentences_second)
            word_count_second_new = sum(len(sentence.split()) for sentence in sentences_second)
            # print(f"second try initiated, word counts - orig: {word_count_second}, new: {word_count_second_new}")

            if word_count_second_new >= 140 and word_count_second_new <= 150:
                return truncated_text_second + "." if truncated_text_second[-1] != "." else truncated_text_second
            if word_count < 100:
                return text
            if word_count < 140:
                return text

        if word_count >= 140 and word_count <= 150:
            return truncated_text + "." if truncated_text[-1] != "." else truncated_text

    # All sentences are complete
    word_count = len(text.split())
    return text