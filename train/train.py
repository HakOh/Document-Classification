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

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec, LineSentence

import tensorflow as tf
from tensorflow.python.client import device_lib
import keras

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
###################### function ######################
def load_doc(filename):
    file = open(filename , 'r',encoding='utf-8', errors='replace')
    text = file.read()
    file.close()
    return text

# vocab 정리

# turn a doc into clean tokens
def clean_doc2(doc, vocab):
    tokens = doc.split()
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens

# load all docs in a directory
def process_docs2(directory, vocab, is_trian):
    documents = []
    for filename in listdir(directory):
        if is_trian and filename.startswith('cv09'):
            continue
        if not is_trian and not filename.startswith('cv09'):
            continue
        path = directory + '/' + filename
        doc = load_doc(path)
        tokens = clean_doc2(doc, vocab)
        if len(tokens.split()) > 1000:
            continue
        if len(tokens.split()) == 0:
            continue
        documents.append(tokens)
    return documents

# turn a doc into clean tokens

def load_doc3(filename):
    file = open(filename, 'r',encoding= 'UTF8', errors = 'replace')
    text = file.readlines()
    file.close()
    return text


def doc_to_clean_lines3(doc, vocab):
    text = doc
    vocab_lines = []

    for i in range(len(text)):
        line_list = text[i].split()
        line_list = [j for j in line_list if j in vocab]
        line_str = " ".join(line_list)
        if line_str:
            vocab_lines.append(line_str)

    return vocab_lines

# load all docs in a directory

def process_docs3(directory, vocab, is_trian):
    lines = []
    for filename in listdir(directory):
        if is_trian and filename.startswith('cv09'):
            continue
        if not is_trian and not filename.startswith('cv09'):
            continue
        path = directory + '/' + filename
        # load and clean the doc
        doc = load_doc3(path)
        doc_lines = doc_to_clean_lines3(doc, vocab)
        # add lines to list
        lines += doc_lines
    return lines

#wiki sentence

def make_wiki_sentences(filename):
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()[:]
        wiki_sentences = []
        for line in lines:
            parts = line.split()
            wiki_sentences.append(parts)
        return wiki_sentences

# load embedding as a dict

def load_embedding(filename):
    file = open(filename, 'r', encoding='utf-8', errors='replace')
    lines = file.readlines()[1:]
    file.close()
    embedding = dict()
    for line in lines:
        parts = line.split()
        print(parts)
        embedding[parts[0]] = asarray(parts[1:], dtype='float32')
    return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    vocab_size = len(vocab) + 1
    weight_matrix = zeros((vocab_size, 300))
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word)
    return weight_matrix


if __name__ == "__main__":

    # load the vocabulary

    vocab_filename = 'vocab8.txt'
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)

    # load all training reviews

    adm_docs = process_docs2('data/adm', vocab, True)
    eco_docs = process_docs2('data/eco', vocab, True)
    edu_docs = process_docs2('data/edu', vocab, True)
    wmn_docs = process_docs2('data/wmn', vocab, True)
    job_docs = process_docs2('data/job', vocab, True)
    wel_docs = process_docs2('data/wel', vocab, True)
    lei_docs = process_docs2('data/lei', vocab, True)
    saf_docs = process_docs2('data/saf', vocab, True)

    print("adm_docs:", len(adm_docs), "eco_docs:",len(eco_docs), "edu_docs:",len(edu_docs), "wmn_docs:",len(wmn_docs), "job_docs:",len(job_docs), "wel_docs:",len(wel_docs),"lei_docs:",len(lei_docs), "saf_docs:",len(saf_docs))

    train_docs = adm_docs + eco_docs + edu_docs + wmn_docs + job_docs + wel_docs + lei_docs + saf_docs

    print("train_docs:", len(train_docs))

    #create the tokenizer
    tokenizer = Tokenizer()
    #fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs)
    #test
    print(train_docs[0])

    #sequence encode
    encoded_docs = tokenizer.texts_to_sequences(train_docs)

    print(len(train_docs[0].split()))

    # pad sequences
    # sentence 크기 맞춰주기 위해 padding

    max_length = max([len(s.split()) for s in train_docs])
    Xtrain = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post')

    print(max_length)

    #one-hot-encoding (optional)
    ytrain = array([[1,0,0,0,0,0,0,0] for _ in range(8582)] + [[0,1,0,0,0,0,0,0] for _ in range(8782)]+ [[0,0,1,0,0,0,0,0] for _ in range(8646)]+ [[0,0,0,1,0,0,0,0] for _ in range(8846)] + [[0,0,0,0,1,0,0,0] for _ in range(8155)] + [[0,0,0,0,0,1,0,0] for _ in range(8261)] + [[0,0,0,0,0,0,1,0] for _ in range(7996)] + [[0,0,0,0,0,0,0,1] for _ in range(8276)])
    ytrain.shape

    print ('Total training: %d' % len(ytrain), len(Xtrain))

    #load all test file

    adm_docs1 = process_docs2('data/adm', vocab, False)
    eco_docs1 = process_docs2('data/eco', vocab, False)
    edu_docs1 = process_docs2('data/edu', vocab, False)
    wmn_docs1 = process_docs2('data/wmn', vocab, False)
    job_docs1 = process_docs2('data/job', vocab, False)
    wel_docs1 = process_docs2('data/wel', vocab, False)
    lei_docs1 = process_docs2('data/lei', vocab, False)
    saf_docs1 = process_docs2('data/saf', vocab, False)

    test_docs = adm_docs1 + eco_docs1 + edu_docs1 + wmn_docs1 + job_docs1 + wel_docs1 + lei_docs1 + saf_docs1

    print("adm_docs:", len(adm_docs1), "eco_docs:",len(eco_docs1), "edu_docs:",len(edu_docs1), "wmn_docs:",len(wmn_docs1), "job_docs:",len(job_docs1), "wel_docs:",len(wel_docs1), "lei_docs:",len(lei_docs1), "saf_docs:",len(saf_docs1))

    #sequence encode
    encoded_docs = tokenizer.texts_to_sequences(test_docs)

    #pad sequences
    Xtest = pad_sequences(encoded_docs, maxlen = max_length, padding='post')

    #define test labels
    # ytest = array([0 for _ in range(100)] + [1 for _ in range(100)]+ [2 for _ in range(900)]+ [3 for _ in range(100)]+ [4 for _ in range(100)]+ [5 for _ in range(100)])
    ytest = array([[1,0,0,0,0,0,0,0] for _ in range(923)] + [[0,1,0,0,0,0,0,0] for _ in range(972)]+ [[0,0,1,0,0,0,0,0] for _ in range(1000)]+ [[0,0,0,1,0,0,0,0] for _ in range(999)] + [[0,0,0,0,1,0,0,0] for _ in range(872)] + [[0,0,0,0,0,1,0,0] for _ in range(892)] + [[0,0,0,0,0,0,1,0] for _ in range(894)] + [[0,0,0,0,0,0,0,1] for _ in range(900)])
    print(Xtest.shape)

    print ('Total training: %d' % len(ytest), len(Xtest))

    # define vocabulary size (largest integer value)
    vocab_size = len(tokenizer.word_index) + 1

    print (vocab_size)

    # load the vocabulary
    vocab_filename = 'vocab8.txt'
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)

    #load training data

    adm_lines = process_docs3('data/adm', vocab, False)
    eco_lines = process_docs3('data/eco', vocab, False)
    edu_lines = process_docs3('data/edu', vocab, False)
    wmn_lines = process_docs3('data/wmn', vocab, False)
    job_lines = process_docs3('data/job', vocab, False)
    wel_lines = process_docs3('data/wel', vocab, False)
    lei_lines = process_docs3('data/lei', vocab, False)
    saf_lines = process_docs3('data/saf', vocab, False)

    sentences = adm_lines + eco_lines + edu_lines + wmn_lines + job_lines + wel_lines + lei_lines + saf_lines
    sentences = list(map(lambda x: x.split(' '), sentences))
    print(sentences[:10])

    print('Total training sentences: %d' % len(sentences))

    wiki_sentences = make_wiki_sentences("wiki_corpus.txt")
    wiki_sentences[:100]

    #원래 데이터 word2vec 후 wiki까지 덮어씌움

    data_sentences = np.array(sentences)
    wiki_sentences = np.array(wiki_sentences)

    gensim_model = Word2Vec(data_sentences, size = 300, window = 5, workers = 8, min_count = 3, iter = 15, sg = 1)
    gensim_model.save("word2vec_8.bin")

    new_model = Word2Vec.load("word2vec_8.bin")
    new_model.train(wiki_sentences, total_examples=new_model.corpus_count, epochs=new_model.iter)

    new_model.wv.vocab

    filename = 'word2vec_8.txt'
    new_model.wv.save_word2vec_format(filename, binary=False)

    # load embedding from file
    raw_embedding = load_embedding('word2vec_8.txt')
    # get vectors in the right order
    embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
    #create the embedding layer
    embedding_layer = Embedding(vocab_size, 300, weights=[embedding_vectors], input_length=max_length, trainable = False)

    print(device_lib.list_local_devices())

    import keras.backend.tensorflow_backend as K
    with K.tf.device('/gpu:0'):
        filter_sizes = [3,4,5]
        drop_out_rate = 0.7
        hidden_dims = 50

        model = Sequential()
        model.add(Embedding(vocab_size, 300, input_length=max_length))
        for idx, sz in enumerate(filter_sizes):
            model.add(Conv1D(filters=100, kernel_size=sz, activation='relu', strides=1, padding='valid'))
            model.add((MaxPooling1D(pool_size=2)))


        model.add(Dropout(drop_out_rate))
        model.add(Flatten())
        model.add(Dense(hidden_dims, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(8, activation='softmax'))
        print(model.summary())

    #compile network
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(Xtrain, ytrain, epochs = 10, verbose = 2)

    #evaluate
    loss, acc = model.evaluate(Xtest, ytest, verbose = 0)
    print('Test Acurracy: %f' %(acc * 100))
