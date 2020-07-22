import os
import argparse
import time

from dataloader import get_data_loader
from utils import seq2sen
from model import Encoder, Decoder

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("results/log.txt", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def save_checkpoint(model, path):
    model_state = {
        'state_dict' : model.state_dict()
    }
    
    torch.save(model_state, path)
    print('A check point has been generated : ' + path)

def main(args):
    # constant definition
    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    a_dim = 512
    h_dim = 512
    attn_dim = 512
    embed_dim = 512
    regularize_constant = 1. # lambda * L

    vocabulary = torch.load(args.voca_path)
    vocab_size = len(vocabulary)
    
    device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu")

    if not args.test:
        # train
        train_loader = get_data_loader(args.path, 'train', args.token_path, args.voca_path, args.batch_size, pad_idx)
        valid_loader = get_data_loader(args.path, 'valid', args.token_path, args.voca_path, args.batch_size, pad_idx)

        encoder = Encoder().to(device)
        decoder = Decoder(a_dim, h_dim, attn_dim, vocab_size, embed_dim).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)
        optimizer = torch.optim.Adam(decoder.parameters(), lr = 0.0001)

        print('Start training ...')
        for epoch in range(args.epochs):
            start_epoch = time.time()
            i = 0

            for src_batch, tgt_batch in train_loader:
                batch_start = time.time()

                src_batch = src_batch.to(device)
                trg_batch = torch.tensor(tgt_batch).to(device)
                
                trg_input = trg_batch[:,:-1] 
                trg_output = trg_batch[:,1:].contiguous().view(-1)
                
                a = encoder(src_batch)
                preds, alphas = decoder(a, trg_input) # [batch, C, vocab_size], [batch, C, L]

                optimizer.zero_grad()

                loss = criterion(preds.view(-1, preds.size(-1)), trg_output) # NLL loss
                regularize_term = regularize_constant * ((1. - torch.sum(alphas, dim = 1)) ** 2).mean()
                
                loss += regularize_term
                loss.backward()

                optimizer.step()

                i = i+1
                
                # flush the GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                batch_time = time.time() - batch_start
                print('[%d/%d][%d/%d] train loss : %.4f | time : %.2fs'%(epoch+1, 100, i, train_loader.size//args.batch_size + 1, loss.item(), batch_time))
                
            epoch_time = time.time() - start_epoch
            print('Time taken for %d epoch : %.2fs'%(epoch+1, epoch_time))

            save_checkpoint(decoder, 'checkpoints/epoch_%d_'%(epoch+1))

        print('End of the training')
        save_checkpoint(decoder, 'checkpoints/final')
    else:
        if os.path.exists(args.checkpoint):
            model_checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(model_checkpoint['state_dict'])
            print("trained model " + args.checkpoint + "is loaded")

        # test
        test_loader = get_data_loader(args.path, 'test', args.token_path, args.voca_path, args.batch_size, pad_idx)
        vocabulary = torch.load(args.voca_path)

        j = 0
        pred = []
        for src_batch, trg_batch in test_loader:
            # predict pred_batch from src_batch with your model.
            # every sentences in pred_batch should start with <sos> token (index: 0) and end with <eos> token (index: 1).
            # every <pad> token (index: 2) should be located after <eos> token (index: 1).
            # example of pred_batch:
            # [[0, 5, 6, 7, 1],
            #  [0, 4, 9, 1, 2],
            #  [0, 6, 1, 2, 2]]

            src_batch = torch.tensor(src_batch).to(device)
            trg_batch = torch.tensor(trg_batch).to(device)
            
            max_length = trg_batch.size(1)
            
            pred_batch = torch.zeros(args.batch_size, 1, dtype = int).to(device) # [batch, 1] = [[0],[0],...,[0]]
            
            # eos_mask[i] = 1 means i-th sentence has eos
            eos_mask = torch.zeros(args.batch_size, dtype = int)
            
            h = listener(src_batch)
            
            for k in range(max_length):
                start = time.time()
                output = speller(pred_batch, h) # [batch, k+1, num_class]

                # greedy search
                output = torch.argmax(F.softmax(output, dim = -1), dim = -1) # [batch_size, k+1]
                predictions = output[:,-1].unsqueeze(1)
                pred_batch = torch.cat([pred_batch, predictions], dim = -1)

                for i in range(args.batch_size):
                    if predictions[i] == eos_idx:
                        eos_mask[i] = 1

                # every sentence has eos
                if eos_mask.sum() == args.batch_size :
                    break
                    
                t = time.time() - start
                print("[%d/%d][%d/%d] prediction done | time : %.2fs"%(j, test_loader.size // args.batch_size + 1, k+1, max_length, t))
            j += 1

            # flush the GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[%d/%d] prediction done"%(j, test_loader.size // args.batch_size + 1))
            pred += seq2sen(pred_batch.cpu().numpy().tolist(), vocabulary)
            ref += seq2sen(trg_batch.cpu().numpy().tolist(), vocabulary)

        with open('results/pred.txt', 'w') as f:
            for line in pred:
                f.write('{}\n'.format(line))

        with open('results/ref.txt', 'w') as f:
            for line in ref:
                f.write('{}\n'.format(line))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LAS')
    parser.add_argument(
        '--path',
        type=str,
        default='data/')

    parser.add_argument(
        '--token_path',
        type=str,
        default='preprocessed_data/tokens')

    parser.add_argument(
        '--voca_path',
        type=str,
        default='preprocessed_data/vocabulary')

    parser.add_argument(
        '--epochs',
        type=int,
        default=100)

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32)

    parser.add_argument(
        '--test',
        action='store_true')

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/final'
    )

    args = parser.parse_args()
    sys.stdout = Logger()

    main(args)