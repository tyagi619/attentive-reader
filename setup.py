'''
This script downloads the SQuAD 2.0 dataset,
GLoVe 300D word vectors and also preprocesses the
dataset.
1- Download GLoVe 300D word vectors
2- Download SQuAD 2.0
3- Pre-process the dataset and word vectors
'''

# TODO - import required libraries
from cmath import inf, sqrt
from urllib.request import urlretrieve
from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile
from subprocess import run
from collections import Counter
import spacy
import ujson as json

import numpy as np


def _get_filename_from_url(download_url):
    filename = download_url.split('/')[-1]
    return filename


def _download(download_url, save_path):
    # TODO - raise error if download fails

    # define a custom class inheriting tqdm to display
    # progress bar while downloading file
    class DownloadProgressBar(tqdm):
        def update_bar(self, n_blocks, b_size, t_size=None):
            if t_size is not None:
                self.total = t_size
            self.update(n_blocks * b_size - self.n)

    with DownloadProgressBar(unit='B',
                             unit_scale=True,
                             miniters=1,
                             desc=download_url.split('/')[-1]) as t:
        filename, headers = urlretrieve(url=download_url,
                                        filename=save_path,
                                        reporthook=t.update_bar)

    return None


def download(download_url_list, output_dir):
    for name, download_url in download_url_list:
        filename = _get_filename_from_url(download_url)
        filePath = output_dir/filename

        if not filePath.exists():
            print(f'Downloading {name}...')
            _download(filePath, str(filePath))

        if filePath.exists() and (filePath.suffix=='.zip'):
            extracted_filename = filename.replace('.zip','')
            extracted_filePath = output_dir/extracted_filename
            if not extracted_filePath.exists():
                print(f'Extracting {filename}...')
                with ZipFile(str(filePath)) as zip_file:
                    zip_file.extractall(str(extracted_filePath))

    # download spacy english language model (only tokenizer is required)
    print(f'Download English language model for spacy...')
    run(['python3','-m','spacy','download','en'])

    return None


def _word_tokenizer(text):
    tokens = nlp(text)
    # TODO - Experiment with using lemma instead of exact word
    # To do this, you will need to load english language pipeline
    # instead of simple language model (which only performs tokenization)
    return [token.text.lower().strip() for token in tokens]


def _convert_token_to_span(text, tokens):
    ptr = 0
    spans = []
    for token in tokens:
        ptr = text.find(token, ptr)
        if ptr < 0:
            raise Exception(f'token {token} not found in {text}')
        spans.append((ptr, ptr+len(token)))
        ptr += len(token)
    return spans


def _build_features(out_file, examples, dataset_type, word2idx, char2idx, is_test=False):
    # In the model, (batch_size, seq_len) is the effective
    # batch size. Thus, all sequences must have same length 
    # with a given dataset type. Across different dataset 
    # type, the seq_len can vary though.
    para_limit = 1000 if is_test else 400
    ques_limit = 100 if is_test else 50
    ans_limit = 30
    char_limit = 16 

    def is_answerable(ex):
        return len(ex['y1s']) > 0 and len(ex['y2s']) > 0

    def drop_example(ex):
        if is_test:
            drop = False
        else:
            drop = len(ex['context_tokens']) > para_limit or \
                   len(ex['ques_tokens']) > ques_limit or \
                   (is_answerable(ex) and
                   ex['y2s'][-1] - ex['y1s'][-1] > ans_limit)
        return drop

    def _get_word_idx(word):
        if word in word2idx:
            return word2idx[word]
        # return OOV if word is not found
        return 1

    def _get_char_idx(char):
        if char in char2idx:
            return char2idx[char]
        # return OOV if char is not found
        return 1    

    total_ = 0
    total = 0
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []

    print(f"Converting {dataset_type} examples to indices...")

    for example in tqdm(examples):
        total_ += 1

        if drop_example(example):
            continue

        total += 1
        
        context_idx = np.zeros(para_limit, dtype=np.int32)
        context_char_idx = np.zeros(char_limit, dtype=np.int32)
        ques_idx = np.zeros(ques_limit, dtype=np.int32)
        ques_char_idx = np.zeros(char_limit, dtype=np.int32)

        for i,token in enumerate(example['context_tokens']):
            context_idx[i] = _get_word_idx(token)
        context_idxs.append(context_idx)

        for i, token in enumerate(example['ques_tokens']):
            ques_idx[i] = _get_word_idx(token)
        ques_idxs.append(ques_idx)

        for i, token in enumerate(example['context_chars']):
            for j, char in enumerate(token):
                if j >= char_limit:
                    break
                context_char_idx[j] = _get_char_idx(char)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example['ques_chars']):
            for j, char in enumerate(token):
                if j >= char_limit:
                    break
                ques_char_idx[j] = _get_char_idx(char)
        ques_char_idxs.append(ques_char_idx)

        if is_answerable(example):
            start, end = example['y1s'][-1], example['y2s'][-1]
        else:
            start, end = -1, -1

        y1s.append(start)
        y2s.append(end)
        ids.append(example['id'])

    np.savez(out_file,
             context_idxs=np.array(context_idxs),
             context_char_idxs=np.array(context_char_idxs),
             ques_idxs=np.array(ques_idxs),
             ques_char_idxs=np.array(ques_char_idxs),
             y1s=np.array(y1s),
             y2s=np.array(y2s),
             ids=np.array(ids))
    print(f"Built {total} / {total_} instances of features in total")
    meta = {'total': total}
    return meta


def _get_embedding(counter, emb_type, emb_file, vec_size, limit=-1):
    assert vec_size is not None, 'Embedding vector size cannot be None'

    embedding_dict = {}
    if emb_file is not None:
        with open(emb_file, 'r') as f:
            line = f.readline()
            arr = line.split()
            token = ''.join(arr[:-vec_size]).lower().strip()
            embed_vec = list(map(float, arr[-vec_size:]))
            if token in counter and counter[token] > limit:
                embedding_dict[token] = embed_vec
    else:
        for token, count in counter.items():
            if count > limit:
                # Initialize the parameters as std = sqrt(2/in_features)
                # This is the standard initialization so that std does not
                # increase or decrease too much when propagated across
                # layers
                embedding_dict[token] = np.random.normal(scale=sqrt(2)/sqrt(vec_size),
                                                         size=(vec_size,)).tolist() 

    print(f"{len(embedding_dict)} tokens have corresponding {emb_type} embedding vector")

    token2idx = {token: idx for idx, token in enumerate(embedding_dict.keys(),2)}

    NULL = '--NULL--'
    OOV = '--OOV--'
    token2idx[NULL] = 0
    token2idx[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]

    emb_mat = [embedding_dict[token2idx[idx]] for idx in range(len(token2idx))]

    return emb_mat, token2idx


def _process_file(file, dataset_type, word_counter, char_counter):
    print(f"Pre-processing {dataset_type} examples...")
    with open(file, 'r') as f:
        source = json.load(f)

    examples = []
    # use this to store exact question, context and answer texts
    # This can come in handy when debugging. The key is the question
    # ID in the processed dataset and value is the dict of context,
    # question, answer and uuid
    eval_examples = {}
    total = 0
    for document in tqdm(source['data']):
        for para in document['paragraphs']:
            context_text = para['context'].replace(
                "''",'" ').replace("``",'" ')
            context_tokens = _word_tokenizer(context_text)
            context_chars = [list(token) for token in context_tokens]
            token_spans = _convert_token_to_span(context_text, context_tokens)

            if dataset_type == 'train':
                for tk in context_tokens:
                    word_counter[tk] += len(para['qas'])
                    for ch in tk:
                        char_counter[ch] += len(para['ques'])

            for ques in para['qas']:
                total += 1
                ques_text = ques['question'].replace(
                    "''",'" ').replace("``",'" ')
                ques_tokens = _word_tokenizer(ques_text)
                ques_chars = [list(token) for token in ques_tokens]

                if dataset_type == 'train':
                    for tk in ques_tokens:
                        word_counter[tk] += 1
                        for ch in ques_chars:
                            char_counter[ch] += 1

                ans_starts = []
                ans_ends = []
                ans_texts = []    
                for ans in ques['answers']:
                    ans_text = ans['text']
                    ans_start = ans['answer_start']
                    ans_end = ans_start + len(ans_text)

                    ans_tk_start = len(context_tokens) + 1
                    ans_tk_end = -1
                    for i, span in enumerate(token_spans):
                        if not (ans_end <= span[0] or ans_start >= span[1]):
                            ans_tk_start = min(ans_tk_start, i)
                            ans_tk_end = max(ans_tk_end, i)
                    
                    if ans_tk_start > ans_tk_end:
                        raise Exception(f'answer for {ques["id"]} does not exist within context')

                    ans_starts.append(ans_start)
                    ans_ends.append(ans_end)
                    ans_texts.append(ans_text)

                example = {'context_tokens': context_tokens,
                           'context_chars': context_chars,
                           'ques_tokens': ques_tokens,
                           'ques_chars': ques_chars,
                           'y1s': ans_starts,
                           'y2s': ans_ends,
                           'id': total
                           }
                exact_example = {'context': context_text,
                                 'question': ques_text,
                                 'answer': ans_texts,
                                 'uuid': ques['id']
                                 }
                examples.append(example)
                eval_examples[str(total)] = exact_example

    return examples, eval_examples


def pre_process(data_dir):
    # TODO - initialize word and char Counter
    word_counter = Counter()
    char_counter = Counter()
    # TODO - process train file into tokens and question-answer
    train_file = str(data_dir/'train-v2.0.json')
    train_examples, train_eval_examples = _process_file(train_file, 'train', word_counter, char_counter)
    # TODO - load word embeddings and word2idx
    glove_file = str(data_dir/'glove.840B.300d'/'glove.840B.300d.txt')
    word_emb_mat, word2idx = _get_embedding(word_counter, 'word',
                                            emb_file=glove_file,
                                            vec_size=300)
    # TODO - load char embeddings and char2idx
    char_emb_mat, char2idx = _get_embedding(char_counter, 'char',
                                            emb_file=None,
                                            vec_size=64)
    # TODO - process SQuAD dataset into indexes
    # TODO - load char embeddings

    pass


if __name__=='__main__':
    download_urls = [('GLoVe 300D', 'https://nlp.stanford.edu/data/glove.840B.300d.zip'),
                     ('SQuAD 2.0 train set', 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'),
                     ('SQuAD 2.0 dev set', 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json')
                    ]
    data_dir = Path('./data')

    download(download_urls, data_dir)

    # load spacy english language model
    nlp = spacy.blank('en')
    # TODO - preprocess and save SQuAD and GLoVe

    pass