from pathlib import Path

from utils import preprocess
import torch.utils.data as Data

root = Path('.')


def load_data(dataset, emb_dim, max_length):
    # Load data
    print("Loading data...")
    datasets = [str(root / 'data' / dataset / s)
        for s in ['train.ss', 'dev.ss', 'test.ss']
    ]
    embedding_filename = 'data/embedding_imdb_yelp13_elc_cd_clt.txt'
    # if dataset in ['yelp13', 'imdb']:
    #     embedding_filename = 'data/embedding_imdb_yelp13.txt'
    # elif dataset in ['cd', 'elc']:
    #     embedding_filename = 'data/embedding_cd_elc.txt'
    print(embedding_filename)
    datasets, embedding, wrd_dict = preprocess.build_dataset(datasets, embedding_filename, emb_dim, max_length)
    trainset, devset, testset = datasets
    # trainlen, devlen, testlen = lengths
    # if shuffle_train:
    #     trainset = trainset.shuffle(30000)
    # devset = devset
    # testset = testset
    # if batch_size != 1:
    #     trainset = trainset.batch(batch_size)
    #     devset = devset.batch(batch_size)
    #     testset = testset.batch(batch_size)
    print("Data loaded.")
    return embedding, trainset, devset, testset, wrd_dict

def loader_set(dataset,batch_size):
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    return loader