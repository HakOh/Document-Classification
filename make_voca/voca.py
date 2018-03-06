from collections import namedtuple
from konlpy.tag import Komoran
from string import punctuation
from pprint import pprint
from sklearn.linear_model import LogisticRegression
import multiprocessing
import numpy as np
import pickle
import os
import re
from collections import Counter
###################### function ######################

# 형태소 분석과 문장 cleaning

# file load
def load_doc(filename):
    file = open(filename , 'r',encoding='utf-8', errors='replace')
    text = file.read()
    file.close()
    return text
# 불러온 파일을 한글만 남기고 다지우고 형태소 분석을 합니다
def clean_doc(doc):
    token = doc.split()
    hangul = re.compile('[^ ㄱ-ㅣ가-힣\n]+')
    tokens = hangul.sub('', doc)
    tokens = re.sub(r'\s{3,}', '',tokens)
    tokens = tokens.split()
    komoran_token = []
    for i in range(len(tokens)):
        dic = komoran.morphs(tokens[i])
        dicSize = len(dic)
        for idx in range(dicSize):
            komoran_token.append(dic[idx])
    tokens = [word for word in komoran_token if len(word) > 1]

    return tokens
# vocab 생성

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_trian and filename.startswith('cv09'):
            continue
        if not is_trian and not filename.startswith('cv09'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # add doc to vocab
        add_doc_to_vocab(path, vocab)

# save list to file
def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    file = open(filename, 'w', encoding="utf-8")
    file.write(data)
    file.close()



if __name__ == "__main__":

    komoran = Komoran()

    # 가장 많이 나오는 word 찾기

    # define vocab
    vocab = Counter()
    # add all docs to vocab
    process_docs('data/adm', vocab, True)
    process_docs('data/eco', vocab, True)
    process_docs('data/edu', vocab, True)
    process_docs('data/wmn', vocab, True)
    process_docs('data/job', vocab, True)
    process_docs('data/wel', vocab, True)
    process_docs('data/lei', vocab, True)
    process_docs('data/saf', vocab, True)


    # print the size of the vocab
    print(len(vocab))
    # print the top words in the vocab
    print(vocab.most_common(30))

    # keep tokens with a minn occurrence

    min_occurane = 5
    tokens = [k for k,c in vocab.items() if c >= min_occurane]
    print(len(tokens))

    # save tokens to a vocabulary file
    save_list(tokens, 'vocab8.txt')
