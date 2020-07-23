import os
import argparse
import time

from dataloader import get_train_data_loader, get_test_data_loader
from utils import seq2sen
from model import Encoder, Decoder
from nltk.translate.bleu_score import corpus_bleu

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
    regularize_constant = 1. # lambda * L => lambda = 1/L
    
    validation_term = 1
    best_bleu = 0.

    vocabulary = torch.load(args.voca_path)
    vocab_size = len(vocabulary)
    
    device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu")
    encoder = Encoder().to(device)
    decoder = Decoder(a_dim, h_dim, attn_dim, vocab_size, embed_dim).to(device)

    # We do not train the encoder
    encoder.eval()

    if not args.test:
        # train
        train_loader = get_train_data_loader(args.path, args.token_path, args.voca_path, args.batch_size, pad_idx)
        valid_loader = get_test_data_loader(args.path, args.token_path, args.voca_path, args.batch_size, pad_idx, dataset_type = 'valid')

        criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)
        optimizer = torch.optim.Adam(decoder.parameters(), lr = 0.0001)

        print('Start training ...')
        for epoch in range(args.epochs):
            start_epoch = time.time()
            i = 0

            ############################################################################################################################################
            # training
            decoder.train()
            for src_batch, trg_batch in train_loader:
                batch_start = time.time()

                src_batch = src_batch.to(device)
                trg_batch = torch.tensor(trg_batch).to(device)
                
                trg_input = trg_batch[:,:-1] 
                trg_output = trg_batch[:,1:].contiguous().view(-1)
                
                a = encoder(src_batch)
                preds, alphas = decoder(a, trg_input) # [batch, C, vocab_size], [batch, C, L]

                optimizer.zero_grad()

                loss = criterion(preds.view(-1, preds.size(-1)), trg_output) # NLL loss
                regularize_term = regularize_constant * ((1. - torch.sum(alphas, dim = 1)) ** 2).mean()
                
                total_loss = loss + regularize_term
                total_loss.backward()

                optimizer.step()

                i = i+1
                
                # flush the GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                batch_time = time.time() - batch_start
                print('[%d/%d][%d/%d] train loss : %.4f (%.4f / %.4f) | time : %.2fs'%(epoch+1, args.epochs, i, train_loader.size//args.batch_size + 1, total_loss.item(), loss.item(), regularize_term.item(), batch_time))
                
            epoch_time = time.time() - start_epoch
            print('Time taken for %d epoch : %.2fs'%(epoch+1, epoch_time))

            ############################################################################################################################################
            # validation
            if i % validation_term == 0:
                decoder.eval()
                j = 0
                pred, ref = [], []

                for src_batch, trg_batch in valid_loader:

                    start = time.time()

                    src_batch = src_batch.to(device) # [batch, 3, 244, 244]
                    trg_batch = torch.tensor(trg_batch).to(device) # [batch * 5, C]
                    trg_batch = torch.split(trg_batch, 5)
                    
                    batches = []
                    for i in range(args.batch_size):
                        batches.append(trg_batch[i].unsqueeze(0))
                    
                    trg_batch = torch.cat(batches, dim = 0) # [batch, 5, C]
                    
                    max_length = trg_batch.size(-1)
                    
                    pred_batch = torch.zeros(args.batch_size, 1, dtype = int).to(device) # [batch, 1] = [[0],[0],...,[0]]
                    
                    # eos_mask[i] = 1 means i-th sentence has eos
                    eos_mask = torch.zeros(args.batch_size, dtype = int)
                    
                    a = encoder(src_batch)
                    
                    for k in range(max_length):
                        
                        output, _ = decoder(a, pred_batch) # [batch, k+1, vocab_size]

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
                            
                    # flush the GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    pred += seq2sen(pred_batch.cpu().numpy().tolist(), vocabulary)
                    for i in range(args.batch_size):
                        ref += [seq2sen(trg_batch[i].cpu().numpy().tolist(), vocabulary)]
                    
                    t = time.time() - start
                    j += 1
                    print("[%d/%d] prediction done | time : %.2fs"%(j, valid_loader.size // args.batch_size + 1, t))

                bleu_1 = corpus_bleu(ref, pred, weights = (1./1.,)) * 100
                bleu_2 = corpus_bleu(ref, pred, weights = (1./2., 1./2.,)) * 100
                bleu_3 = corpus_bleu(ref, pred, weights = (1./3., 1./3., 1./3.,)) * 100
                bleu_4 = corpus_bleu(ref, pred, weights = (1./4., 1./4., 1./4., 1./4.,)) * 100

                print(f'BLEU-1: {bleu_1:.2f}')
                print(f'BLEU-2: {bleu_2:.2f}')
                print(f'BLEU-3: {bleu_3:.2f}')
                print(f'BLEU-4: {bleu_4:.2f}')

                if bleu_1 > best_bleu :
                    best_bleu = bleu_1
                    print('Best BLEU-1 has been updated : %.2f'%(best_bleu))
                    save_checkpoint(decoder, 'checkpoints/best')
            
            ################################################################################################################################################################
        print('End of the training')
    else:
        if os.path.exists(args.checkpoint):
            decoder_checkpoint = torch.load(args.checkpoint)
            decoder.load_state_dict(decoder_checkpoint['state_dict'])
            print("trained decoder " + args.checkpoint + " is loaded")

        decoder.eval()

        # test
        test_loader = get_test_data_loader(args.path, args.token_path, args.voca_path, args.batch_size, pad_idx)

        j = 1
        pred, ref = [], []
        for src_batch, trg_batch in test_loader:
            # predict pred_batch from src_batch with your model.
            # every sentences in pred_batch should start with <sos> token (index: 0) and end with <eos> token (index: 1).
            # every <pad> token (index: 2) should be located after <eos> token (index: 1).
            # example of pred_batch:
            # [[0, 5, 6, 7, 1],
            #  [0, 4, 9, 1, 2],
            #  [0, 6, 1, 2, 2]]

            src_batch = src_batch.to(device) # [batch, 3, 244, 244]
            trg_batch = torch.tensor(trg_batch).to(device) # [batch * 5, C]
            trg_batch = torch.split(trg_batch, 5)
            
            batches = []
            for i in range(args.batch_size):
                batches.append(trg_batch[i].unsqueeze(0))
            
            trg_batch = torch.cat(batches, dim = 0) # [batch, 5, C]
            
            max_length = trg_batch.size(-1)
            
            pred_batch = torch.zeros(args.batch_size, 1, dtype = int).to(device) # [batch, 1] = [[0],[0],...,[0]]
            
            # eos_mask[i] = 1 means i-th sentence has eos
            eos_mask = torch.zeros(args.batch_size, dtype = int)
            
            a = encoder(src_batch)
            
            for k in range(max_length):
                start = time.time()
                output, _ = decoder(a, pred_batch) # [batch, k+1, vocab_size]

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

            # flush the GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pred += seq2sen(pred_batch.cpu().numpy().tolist(), vocabulary)
            for i in range(args.batch_size):
                ref += [seq2sen(trg_batch[i].cpu().numpy().tolist(), vocabulary)]
            
            print("[%d/%d] prediction done"%(j, test_loader.size // args.batch_size + 1))

            bleu_1 = corpus_bleu(ref, pred, weights = (1./1.,)) * 100
            bleu_2 = corpus_bleu(ref, pred, weights = (1./2., 1./2.,)) * 100
            bleu_3 = corpus_bleu(ref, pred, weights = (1./3., 1./3., 1./3.,)) * 100
            bleu_4 = corpus_bleu(ref, pred, weights = (1./4., 1./4., 1./4., 1./4.,)) * 100

            print(f'BLEU-1: {bleu_1:.2f}')
            print(f'BLEU-2: {bleu_2:.2f}')
            print(f'BLEU-3: {bleu_3:.2f}')
            print(f'BLEU-4: {bleu_4:.2f}')
            
            j += 1

            with open('results/pred.txt', 'w') as f:
                for line in pred:
                    f.write('{}\n'.format(line))

            with open('results/ref.txt', 'w') as f:
                for lines in ref:
                    for line in lines:
                        f.write('{}\n'.format(line))
                    f.write('_'*50 + '\n')


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
        default=200)

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
        default='checkpoints/best'
    )

    args = parser.parse_args()
    sys.stdout = Logger()

    main(args)