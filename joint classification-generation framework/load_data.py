# -*- coding: utf-8 -*-

def load_data():
    # text
    train_data = []
    dev_data = []
    test_data = []
    train_expand=[]
    dev_expand=[]
    test_expand=[]
    with open('pdtb_explanation/train_explanation.txt','r',encoding='utf-8') as ftrain:
        lines=ftrain.readlines()
        lines=[line.strip() for line in lines]
        for line in lines:
            train_data.append(line)
    with open('pdtb_explanation/dev_explanation.txt','r',encoding='utf-8') as fdev:
        lines=fdev.readlines()
        lines=[line.strip() for line in lines]
        for line in lines:
            dev_data.append(line)
    with open('pdtb_explanation/test.txt','r',encoding='utf-8') as ftest:
        lines=ftest.readlines()
        lines=[line.strip() for line in lines]
        for line in lines:
            test_data.append(line)
    # with open('generate_t5/train.txt','r',encoding='utf-8') as ftrain_expand:
    #     lines=ftrain_expand.readlines()
    #     lines=[line.strip() for line in lines]
    #     for line in lines:
    #         train_expand.append(line)
    # with open('generate_t5/dev.txt','r',encoding='utf-8') as fdev_expand:
    #     lines=fdev_expand.readlines()
    #     lines=[line.strip() for line in lines]
    #     for line in lines:
    #         dev_expand.append(line)
    # with open('generate_t5/test.txt','r',encoding='utf-8') as ftest_expand:
    #     lines=ftest_expand.readlines()
    #     lines=[line.strip() for line in lines]
    #     for line in lines:
    #         test_expand.append(line)
            
    return train_data, dev_data, test_data
