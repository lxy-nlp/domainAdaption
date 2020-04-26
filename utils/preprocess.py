import os
import csv
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
reading_col_name = ['usr', 'prd', 'rating', 'content']


def build_dataset(filenames, embedding_filename, emb_dim, max_length):
    wrd_dict, wrd_index, embedding = load_embedding(embedding_filename,emb_dim)
    allcontent = []
    allrating = []
    dataset = []
    for filename in filenames:
        dataframe = pd.read_csv(filename, sep='\t\t', names=reading_col_name, engine='python')
        content = dataframe['content'].tolist()
        rating = dataframe['rating'].tolist()
        # 将　评论内容和评分等级（标签）　提取出来,　将单词转换成词向量的索引
        for i, con in enumerate(content):
            content_index = sentence_transform(con, wrd_index, max_length)
            content[i] = content_index
        allcontent.append(content)
        allrating.append(rating)

    # 装载数据集
    for content, rating in zip(allcontent, allrating):
        content_X = torch.tensor(content)
        content_Y = torch.tensor(rating)
        set = Data.TensorDataset(content_X,content_Y)
        dataset.append(set)

    return dataset, embedding.values, wrd_dict


#       将句子中的单词转换成索引
def sentence_transform(doc,word_index,max_length):
    sentences = doc.split('<sssss>')
    str = ''
    for sen in sentences:
        str += ' '.join(sen.split())
    sentence_word = str.split(' ')
    sentence_index = np.zeros((max_length,), dtype=np.int)
    i = 0
    for word in sentence_word:
        if i == max_length:
            break
        if word in word_index:
            sentence_index[i] = word_index[word]
            i += 1
    return sentence_index


# load an embedding file
def load_embedding(filename, emb_dim):
    try:
        emb_col_name = ['wrd'] + [i for i in range(emb_dim + 1)]
        data_frame = pd.read_csv(
            filename, sep=' ', header=0, names=emb_col_name)
    except pd.errors.ParserError:
        emb_col_name = ['wrd'] + [i for i in range(emb_dim)]
        data_frame = pd.read_csv(
            filename, sep=' ', header=0, names=emb_col_name)
    data_frame = data_frame.sort_values('wrd')
    embedding = data_frame[emb_col_name[1: emb_dim + 1]]
    wrd_dict = data_frame['wrd'].tolist()
    wrd_index = {s: i for i, s in enumerate(wrd_dict)}
    return wrd_dict, wrd_index, embedding


