import os, argparse
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import sen2seq

class ImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolder, self).__getitem__(index)
        return (original_tuple + (self.imgs[index][0],))

class Flickr8k_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokens, voca):
        self.data = data
        self.trg = tokens
        self.voca = voca

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx_type = str(type(idx))
        caption_idx = np.random.randint(5)
        if idx_type == "<class 'int'>":
            image, _, path = self.data[idx] # image : [3, 224, 224]
            x = image # [3, 244, 244]
            y = [sen2seq(self.voca, x[1:], 0, 1) for x in self.trg if path.split('/')[3] in x[0]]

        elif idx_type == "<class 'slice'>":
            x, y = None, None
            
            for i in range(*idx.indices(len(self.data))):
                image, _, path = self.data[i] # image : [3, 224, 224]
                if x is None:
                    x = image.unsqueeze(0) # [1, 3, 244, 244]
                    y = [sen2seq(self.voca, x[1:]) for x in self.trg if path.split('/')[3] in x[0]] # [1, 5, caption length]
                else:
                    next_x = image.unsqueeze(0) # [1, 3, 244, 244]
                    next_y = [sen2seq(self.voca, x[1:]) for x in self.trg if path.split('/')[3] in x[0]] # [5, caption length]

                    x = torch.cat([x, next_x], dim = 0) # [(i+1), 3, 244, 244]
                    y += next_y # [(i+1)*5, caption_lengths]
        else:
            raise TypeError('Not compatible type')

        return x, y

class Flickr8k_Dataset_with_Random_Sampling(torch.utils.data.Dataset):
    def __init__(self, data, tokens, voca):
        self.data = data
        self.trg = tokens
        self.voca = voca

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx_type = str(type(idx))

        if idx_type == "<class 'int'>":
            image, _, path = self.data[idx] # image : [3, 224, 224]
            x = image # [3, 244, 244]
            y = [sen2seq(self.voca, x[1:]) for x in self.trg if path.split('/')[3] in x[0]]
            
            y = y[np.random.randint(5)]

        elif idx_type == "<class 'slice'>":
            x, y = [], []
            
            for i in range(*idx.indices(len(self.data))):
                image, _, path = self.data[i] # image : [3, 224, 224]
                captions = [sen2seq(self.voca, x[1:]) for x in self.trg if path.split('/')[3] in x[0]]
                caption = captions[np.random.randint(5)]
                x.append(image.unsqueeze(0))
                y.append(caption)

            x = torch.cat(x, dim = 0)
        else:
            raise TypeError('Not compatible type')

        return x, y

class DataLoader:
    def __init__(self, dataset, batch_size, pad_idx):
        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        self.pad_idx = pad_idx

    def __iter__(self):
        self.index = 0
        return self
    
    def pad(self, batch):
        max_len = 0
        for seq in batch:
            if max_len < len(seq):
                max_len = len(seq)

        for i in range(len(batch)):
            batch[i] += [self.pad_idx] * (max_len - len(batch[i]))

        return batch

    def __next__(self):
        if self.batch_size * self.index >= self.size:
            raise StopIteration

        src_batch, trg_batch = self.dataset[self.batch_size * self.index : self.batch_size * (self.index+1)]
        trg_batch = self.pad(trg_batch)

        self.index += 1

        return src_batch, trg_batch

def get_train_data_loader(data_root, token_path, voca_path, batch_size, pad_idx):

    # load tokens
    if not os.path.exists(token_path):
        tokens = []
        with open(data_root + 'Flickr8k.token.txt', 'r') as reader:
            for line in reader:
                token = line.split()
                if '.' in token :
                    token.remove('.')
                tokens.append(token)
        torch.save(tokens, 'preprocessed_data/tokens')
    else:
        print('tokens are loaded from ' + token_path)
        tokens = torch.load(token_path)

    # load vocabulary
    if not os.path.exists(voca_path):
        vocabulary = {'<sos>' : 0, '<eos>' : 1, '<pad>' : 2}
        idx = 3

        for token in tokens:
            for word in token[1:]:
                word = word.lower()
                if word not in vocabulary:
                    vocabulary[word] = idx
                    idx += 1

        torch.save(vocabulary, 'preprocessed_data/vocabulary')
    else:
        vocabulary = torch.load(voca_path)
        print('vocabulary is loaded from ' + voca_path)

    data = ImageFolder(root=data_root + 'train',
                            transform=transforms.Compose([
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]))
    
    dataset = Flickr8k_Dataset_with_Random_Sampling(data, tokens, vocabulary)

    return DataLoader(dataset, batch_size, pad_idx)

def get_test_data_loader(data_root, token_path, voca_path, batch_size, pad_idx, dataset_type = 'test'):

    # load tokens
    if not os.path.exists(token_path):
        tokens = []
        with open(data_root + 'Flickr8k.token.txt', 'r') as reader:
            for line in reader:
                token = line.split()
                if '.' in token :
                    token.remove('.')
                tokens.append(token)
        torch.save(tokens, 'preprocessed_data/tokens')
    else:
        print('tokens are loaded from ' + token_path)
        tokens = torch.load(token_path)

    # load vocabulary
    if not os.path.exists(voca_path):
        vocabulary = {'<sos>' : 0, '<eos>' : 1, '<pad>' : 2}
        idx = 3

        for token in tokens:
            for word in token[1:]:
                word = word.lower()
                if word not in vocabulary:
                    vocabulary[word] = idx
                    idx += 1

        torch.save(vocabulary, 'preprocessed_data/vocabulary')
    else:
        vocabulary = torch.load(voca_path)
        print('vocabulary is loaded from ' + voca_path)

    data = ImageFolder(root=data_root + dataset_type,
                            transform=transforms.Compose([
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]))
    
    dataset = Flickr8k_Dataset(data, tokens, vocabulary)

    return DataLoader(dataset, batch_size, pad_idx)