# -*- coding: utf-8 -*-    parser.add_argument('--len_arg',      default=512,      type=int, help='Argument length')

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='IDRR')

    # dataset
    parser.add_argument('--train_size',   default=12775,    type=int, help='Train set size')
    parser.add_argument('--dev_size',     default=1183,     type=int, help='Dev set size')
    parser.add_argument('--test_size',    default=1046,     type=int, help='Test set size')
    parser.add_argument('--num_class',    default=4,        type=int, help='Number of classes')

    # # model arguments
    parser.add_argument('--vocab_size',   default=50265,    type=int, help='Size of BERT vocab')
    parser.add_argument('--in_dim',       default=768,      type=int, help='Size of input word vector')
    parser.add_argument('--h_dim',        default=768,      type=int, help='Size of hidden unit')
    parser.add_argument('--len_arg',      default=512,      type=int, help='Argument length')
    parser.add_argument('--arg1_len',     default=300,       type=int, help='Argument_1 max length')


    # # training arguments
    parser.add_argument('--seed',         default=42,      type=int, help='seed for reproducibility')
    parser.add_argument('--batch_size',   default=12,       type=int, help='batchsize for optimizer updates')
    parser.add_argument('--wd',           default=0.01,     type=float, help='weight decay')  # 1e-3

    parser.add_argument('--num_epoch',    default=6,       type=int, help='number of total epochs to run')
    parser.add_argument('--lr',           default=1e-5,     type=float, help='initial learning rate')
    parser.add_argument('--lr_warmup_steps', default=0, type=float, help='initial learning rate')
    parser.add_argument('--warm_ratio',   default=0.1,      type=float, help='ratio of warm up')

    parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate')

    parser.add_argument('--file_out',  default='out',  type=str, help='Result file name')
    
    args = parser.parse_args()

    return args
