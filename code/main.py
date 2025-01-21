# -*- coding: utf-8 -*-
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import f1_score
from load_data import load_data
from prepro_data import prepro_data_train, prepro_data_dev, prepro_data_test
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup,T5Tokenizer
from parameter import parse_args
from tqdm import tqdm
import random

from model import RoBERTa_MLM

torch.cuda.empty_cache()
args = parse_args()  # load parameters


# set seed for random number
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



# random_seed=random.randint(0,10000)
setup_seed(args.seed)
# print(f"randomly generated seed:{random_seed}")
# setup_seed(77694)
# load RoBERTa model
model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
tokenizer_t5=T5Tokenizer.from_pretrained('t5-base')

# load data tsv file
train_data, dev_data, test_data= load_data()

# get arg_1 arg_2 label from data
train_arg_1, train_arg_2, train_label,train_label_conn,train_explanation= prepro_data_train(train_data)
dev_arg_1, dev_arg_2, dev_label,dev_label_conn,dev_explanation= prepro_data_dev(dev_data)
test_arg_1, test_arg_2, test_label,test_label_conn= prepro_data_test(test_data)

label_conn=torch.LongTensor(train_label_conn)
dev_conn=torch.LongTensor(dev_label_conn)
label_tr = torch.LongTensor(train_label)
label_de = torch.LongTensor(dev_label)
label_te = torch.LongTensor(test_label)
print('Data loaded')

ans_word={'although':0,'nevertheless':1,'but':2,'however':3,
          'because':4,'as':5,'so':6,'consequently':7,'thus':8,'since':9,
          'instead':10,'rather':11,'or':12,'and':13,'also':14,'furthermore':15,'instance':16,'example':17,'first':18,'indeed':19,'specifically':20,
          'then':21,'subsequently':22,'previously':23,'earlier':24,'after':25,'meanwhile':26}

Token_id=[1712,21507,53,959,
           142,25,98,28766,4634,187,
           1386,1195,50,8,67,43681,4327,1246,78,5329,4010,
           172,8960,1433,656,71,6637]


def get_batch(text_data1, text_data2, explanation, indices):
    indices_ids = []
    indices_mask = []
    mask_indices = []  # the place of '<mask>' in 'input_ids'
    t5_ids=[]
    t5_mask=[]
    expla_ids=[]
#'Arg:'+text_data1[idx]+'Arg2:'+text_data2[idx]+'</s></s>The relationship between Arg1 and Arg2 is'+'<mask>'
    for idx in indices:
        # first_sentence=prompt[idx].split('The second sentence')[0][5:]
        encode_dict = tokenizer.encode_plus(
            # 'Arg1:' + text_data1[idx] + 'Arg2:' + text_data2[idx] + '</s></s>The conjunction between Arg1 and Arg2 is'+' <mask> '+'.The prompt information is:'+prompt[idx],  #Prompt 2
            # prompt[idx]+'The conjunction between the first sentence and the second sentence is '+'<mask>',
            # text_data1[idx] + ' <mask> ' + text_data2[idx],                 # Prompt 1
            'Arg1:' + text_data1[idx] + '.Arg2:' + text_data2[idx] + '</s></s>The conjunction between Arg1 and Arg2 is ' + '<mask> ',  # Prompt 2
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        indices_ids.append(encode_dict['input_ids'])
        indices_mask.append(encode_dict['attention_mask'])
        encode_dict_t5 = tokenizer_t5.encode_plus(
            'Arg1:' + text_data1[idx] + '.Arg2:' + text_data2[idx] + '</s></s>The conjunction between Arg1 and Arg2 is ' + '<mask>',  # Prompt 2
            # text_data1[idx] + ' <mask> ' + text_data2[idx],  # Prompt 1
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        t5_ids.append(encode_dict_t5['input_ids'])
        t5_mask.append(encode_dict_t5['attention_mask'])
        encode_dict_explanation = tokenizer_t5.encode_plus(
            explanation[idx],
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        # print(encode_dict_explanation['input_ids'])
        expla_ids.append(encode_dict_explanation['input_ids'])
        try: mask_indices.append(np.argwhere(np.array(encode_dict['input_ids']) == 50264)[0][1])  # id of <mask> is 50264
        except IndexError:
            print(encode_dict['input_ids'])
            print(np.argwhere(np.array(encode_dict['input_ids']) == 50264))

    batch_ids = torch.cat(indices_ids, dim=0)
    batch_mask = torch.cat(indices_mask, dim=0)
    batch_t5_ids=torch.cat(t5_ids,dim=0)
    batch_t5_mask=torch.cat(t5_mask,dim=0)
    batch_expla_ids=torch.cat(expla_ids,dim=0)

    return batch_ids, batch_mask, mask_indices,batch_t5_ids,batch_t5_mask,batch_expla_ids

def get_batch_dev(text_data1, text_data2, explanation, prompt,indices):
    indices_ids = []
    indices_mask = []
    mask_indices = []  # the place of '<mask>' in 'input_ids'
    t5_ids=[]
    t5_mask=[]
    expla_ids=[]
#'Arg:'+text_data1[idx]+'Arg2:'+text_data2[idx]+'</s></s>The relationship between Arg1 and Arg2 is'+'<mask>'
    for idx in indices:
        # first_sentence = prompt[idx].split('The second sentence')[0][6:]
        encode_dict = tokenizer.encode_plus(
            # prompt[idx] + 'The conjunction between the first sentence and the second sentence is ' + '<mask>',
            # 'Arg1:' + text_data1[idx] + 'Arg2:' + text_data2[idx] + '</s></s>The conjunction between Arg1 and Arg2 is'+' <mask> '+'.The prompt information is:'+first_sentence,  #Prompt 2
            # text_data1[idx] + ' <mask> ' + text_data2[idx],                 # Prompt 1
            'Arg1:' + text_data1[idx] + 'Arg2:' + text_data2[idx] + '</s></s>The conjunction between Arg1 and Arg2 is ' + '<mask> ',  # Prompt 2
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        indices_ids.append(encode_dict['input_ids'])
        indices_mask.append(encode_dict['attention_mask'])
        encode_dict_t5 = tokenizer_t5.encode_plus(
            'Arg1:' + text_data1[idx] + 'Arg2:' + text_data2[
                idx] + '</s></s>The conjunction between Arg1 and Arg2 is' + ' <mask> ',  # Prompt 2
            # text_data1[idx] + ' <mask> ' + text_data2[idx],  # Prompt 1
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        t5_ids.append(encode_dict_t5['input_ids'])
        t5_mask.append(encode_dict_t5['attention_mask'])
        encode_dict_explanation = tokenizer_t5.encode_plus(
            explanation[idx],
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        # print(encode_dict_explanation['input_ids'])
        expla_ids.append(encode_dict_explanation['input_ids'])
        try: mask_indices.append(np.argwhere(np.array(encode_dict['input_ids']) == 50264)[0][1])  # id of <mask> is 50264
        except IndexError:
            print(encode_dict['input_ids'])
            print(np.argwhere(np.array(encode_dict['input_ids']) == 50264))

    batch_ids = torch.cat(indices_ids, dim=0)
    batch_mask = torch.cat(indices_mask, dim=0)
    batch_t5_ids=torch.cat(t5_ids,dim=0)
    batch_t5_mask=torch.cat(t5_mask,dim=0)
    batch_expla_ids=torch.cat(expla_ids,dim=0)

    return batch_ids, batch_mask, mask_indices,batch_t5_ids,batch_t5_mask,batch_expla_ids

def get_batch_test(text_data1, text_data2,indices):
    indices_ids = []
    indices_mask = []
    mask_indices = []  # the place of '<mask>' in 'input_ids'
    t5_mask=[]
    for idx in indices:
        # first_sentence = prompt[idx].split('The second sentence')[0][6:]
        encode_dict = tokenizer.encode_plus(
            # prompt[idx] + 'The conjunction between the first sentence and the second sentence is ' + '<mask>',
            # 'Arg1:' + text_data1[idx] + 'Arg2:' + text_data2[idx] + '</s></s>The conjunction between Arg1 and Arg2 is'+'<mask>'+'.The prompt information is:'+prompt[idx],  #Prompt 2
            # text_data1[idx] + ' <mask> ' + text_data2[idx],                 # Prompt 1
            'Arg1:' + text_data1[idx] + 'Arg2:' + text_data2[idx] + '</s></s>The conjunction between Arg1 and Arg2 is ' + '<mask>',  # Prompt 2
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        indices_ids.append(encode_dict['input_ids'])
        indices_mask.append(encode_dict['attention_mask'])
        encode_dict_t5=tokenizer_t5.encode_plus(
            'Arg1:' + text_data1[idx] + 'Arg2:' + text_data2[idx] + '</s></s>The conjunction between Arg1 and Arg2 is ' + '<mask>',  # Prompt 2
            # text_data1[idx] + '<mask>' + text_data2[idx],                 # Prompt 1
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        t5_mask.append(encode_dict_t5['attention_mask'])
        try: mask_indices.append(np.argwhere(np.array(encode_dict['input_ids']) == 50264)[0][1])  # id of <mask> is 50264
        except IndexError:
            print(encode_dict['input_ids'])
            print(np.argwhere(np.array(encode_dict['input_ids']) == 50264))

    batch_ids = torch.cat(indices_ids, dim=0)
    batch_mask = torch.cat(indices_mask, dim=0)
    batch_mask_t5 = torch.cat(t5_mask,dim=0)

    return batch_ids, batch_mask, mask_indices,batch_mask_t5


# ---------- network ----------
net = RoBERTa_MLM(args).cuda()
best_mode=torch.load('weights/best_model_30.pth')
net.transformer_layer.load_state_dict(best_mode['transformer'])
# net.transformer_layer1.load_state_dict(best_mode['transformer1'])
# net.transformer_layer2.load_state_dict(best_mode['transformer2'])
# net.transformer_layer3.load_state_dict(best_mode['transformer3'])
# net.transformer_layer4.load_state_dict(best_mode['transformer4'])
net.generate_model.load_state_dict(best_mode['generate_model'])
# AdamW
# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
#     {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]
params=[
    {'params':net.RoBERTa_MLM.parameters(),'lr':5e-6,'weight_decay':0.0001},
    {'params':net.transformer_layer.parameters(),'lr':3e-5,'weight_decay':0.001},
    # {'params':net.transformer_layer1.parameters(),'lr':3e-5,'weight_decay':0.001},
    # {'params':net.transformer_layer2.parameters(),'lr':3e-5,'weight_decay':0.001},
    # {'params':net.transformer_layer3.parameters(),'lr':3e-5,'weight_decay':0.001},
    # {'params':net.transformer_layer4.parameters(),'lr':3e-5,'weight_decay':0.001},
    {'params':net.generate_model.parameters(),'lr':3e-5,'weight_decay':0.001}
]
optimizer = AdamW(params=params)
# optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.lr,weight_decay=args.wd)
# optimizer = Adan(optimizer_grouped_parameters,args.lr)
lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=args.lr_warmup_steps,
                                                num_training_steps=len(train_data) * args.num_epoch)

criterion = nn.CrossEntropyLoss().cuda()

# creat file to save model and result
file_out = open('./' + args.file_out + '.txt', "w")

print('epoch_num:', args.num_epoch)
print('epoch_num:', args.num_epoch, file=file_out)
print('wd:', args.wd)
print('wd:', args.wd, file=file_out)
print('initial_lr:', args.lr)
print('initial_lr:', args.lr, file=file_out)

best_f1=0
##################################  epoch  #################################
for epoch in range(args.num_epoch):
    print('Epoch: ', epoch + 1)
    print('Epoch: ', epoch + 1, file=file_out)
    # all_indices = torch.randperm(args.train_size).split(args.batch_size)
    loss_epoch = 0.0
    loss1_epoch=0.0
    loss2_epoch=0.0
    mutual_loss_epoch=0.0
    acc = 0.0
    f1_pred = torch.IntTensor([]).cuda()
    f1_truth = torch.IntTensor([]).cuda()
    start = time.time()

    # print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])
    # print('lr:', optimizer.state_dict()['param_groups'][0]['lr'], file=file_out)

    ############################################################################
    ##################################  train  #################################
    ############################################################################
    net.train()
    all_indices = torch.randperm(args.train_size).split(args.batch_size)
    for i, batch_indices in enumerate(tqdm(all_indices,desc='Training Batch'), 1):
        # get a batch of wordvecs
        batch_arg, mask_arg, token_mask_indices,batch_t5_ids,batch_t5_mask,batch_expla_ids = get_batch(train_arg_1, train_arg_2, train_explanation, batch_indices)

        batch_arg = batch_arg.cuda()
        mask_arg = mask_arg.cuda()
        batch_t5_mask=batch_t5_mask.cuda()
        batch_expla_ids=batch_expla_ids.cuda()

        y = Variable(label_tr[batch_indices]).cuda()
        y_conn=label_conn[batch_indices].cuda() #[12] 三级标签实际

        # fed data into network
        #out_sense[12,4]一级标签的独热编码  out_ans[12,27] 三级标签的logits
        out_sense, out_ans, out_gen= net(batch_arg, mask_arg, token_mask_indices, Token_id,batch_t5_mask,batch_expla_ids)
        #输入的ids，输入的attention_mask，mask在输入中的位置，三级连接词对应的ids，转折类中三级连接词数量，因果类中三级连接词数量，扩展类中三级连接词数量，顺承类中三级连接词数量
        _, pred_ans = torch.max(out_ans, dim=1) #[12] 三级标签预测

        _, pred = torch.max(out_sense, dim=1) #【12】 一级标签预测
        _, truth = torch.max(y, dim=1) #[12] 一级标签实际

        num_correct = (pred == truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)

        # loss
        loss1 = criterion(out_ans, y_conn)
        loss2 = criterion(out_gen.contiguous().view(-1, out_gen.size(-1)), batch_expla_ids[:, 1:].contiguous().view(-1))
        loss=0.8*loss1+0.2*loss2
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # report
        loss_epoch += loss.item()
        loss1_epoch+=loss1.item()
        loss2_epoch+=loss2.item()
        if i % (3000 // args.batch_size) == 0:
            print('loss={:.4f},classification loss={:.4f},generation loss={:.4f},acc={:.4f}, F1_score={:.4f}'.format(loss_epoch / (3000 // args.batch_size),
                            loss1_epoch / (3000 // args.batch_size),loss2_epoch / (3000 // args.batch_size),acc / 3000,f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                              average='macro')))
            print('loss={:.4f}, acc={:.4f}, F1_score={:.4f}'.format(loss_epoch / (3000 // args.batch_size), acc / 3000,
                                                                    f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                              average='macro')), file=file_out)
            loss_epoch = 0.0
            loss1_epoch=0.0
            loss2_epoch=0.0
            acc = 0.0
            f1_pred = torch.IntTensor([]).cuda()
            f1_truth = torch.IntTensor([]).cuda()
        del batch_arg, mask_arg,batch_t5_mask,batch_expla_ids
    end = time.time()
    print('Training Time: {:.2f}s'.format(end - start))
    torch.cuda.empty_cache()

    ############################################################################
    ##################################  dev  ###################################
    ############################################################################
    all_indices = torch.randperm(args.dev_size).split(args.batch_size)
    loss_epoch = []
    dev_loss1_epoch=0.0
    dev_loss2_epoch=0.0
    acc = 0.0
    f1_pred = torch.IntTensor([]).cuda()
    f1_truth = torch.IntTensor([]).cuda()

    net.eval()
    with torch.no_grad():
        for i, batch_indices in enumerate(tqdm(all_indices, desc='Deving Batch'), 1):
        # for batch_indices in all_indices:
            # get a batch of wordvecs
            batch_arg, mask_arg, token_mask_indices,batch_t5_ids,batch_t5_mask,batch_expla_ids  = get_batch(dev_arg_1, dev_arg_2,dev_explanation, batch_indices)

            batch_arg = batch_arg.cuda()
            mask_arg = mask_arg.cuda()
            batch_expla_ids=batch_expla_ids.cuda()
            batch_t5_mask=batch_t5_mask.cuda()

            y = Variable(label_de[batch_indices]).cuda()
            y_conn = dev_conn[batch_indices].cuda()

            # fed data into network
            out_sense, out_ans, out_gen= net(batch_arg, mask_arg, token_mask_indices, Token_id,batch_t5_mask,batch_expla_ids)
            _, pred = torch.max(out_sense, dim=1)
            _, truth = torch.max(y, dim=1)
            num_correct = (pred == truth).sum()
            acc += num_correct.item()
            f1_pred = torch.cat((f1_pred, pred), 0)
            f1_truth = torch.cat((f1_truth, truth), 0)

            loss1 = criterion(out_ans, y_conn)
            loss2 = criterion(out_gen.contiguous().view(-1, out_gen.size(-1)), batch_expla_ids[:, 1:].contiguous().view(-1))

            dev_loss1_epoch+=loss1.item()
            dev_loss2_epoch+=loss2.item()

            # if i==1:
            #     encoder_output = net.RoBERTa_MLM.roberta(batch_arg, mask_arg)[0].cuda()
            #     encoder_output1=net.transformer_layer(encoder_output)
            #     # encoder_output1=net.fc(encoder_output)
            #     # encoder_output = net.generate_model.encoder(batch_ids_t5).last_hidden_state.cuda()
            #     for j in range(len(batch_arg)):
            #         decoder_input_ids = torch.tensor([[37]]).cuda()
            #         for _ in range(200):
            #             decoder_output = net.generate_model.decoder(decoder_input_ids,
            #                                                         encoder_hidden_states=encoder_output1[j].unsqueeze(
            #                                                             0),
            #                                                         encoder_attention_mask=batch_t5_mask[j].unsqueeze(
            #                                                             0))
            #             logits = net.generate_model.lm_head(decoder_output.last_hidden_state)
            #             next_token_logits = logits[:, -1, :]  # [1,32128]
            #             next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(0)  # [1,1]
            #             decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)
            #         generated_text = tokenizer_t5.decode(decoder_input_ids[0], skip_special_tokens=True)
            #         print('原文本： ', tokenizer_t5.decode(batch_t5_ids[j], skip_special_tokens=True))
            #         print('生成结果:', generated_text)

            del batch_arg, mask_arg,batch_t5_mask,batch_expla_ids,y,y_conn
        print('classification loss={:.4f},generation loss={:.4f} , acc={:.4f}, F1_score={:.4f}'.format(
            dev_loss1_epoch / args.dev_size, dev_loss2_epoch / args.dev_size,acc / args.dev_size,
            f1_score(f1_truth.cpu(), f1_pred.cpu(),
                     average='macro')))
        torch.cuda.empty_cache()
        dev_acc=acc / args.dev_size
        dev_f1=f1_score(f1_truth.cpu(), f1_pred.cpu(),average='macro')
        if dev_f1>best_f1:
            best_f1=dev_f1
            torch.save(
                {
                    'roberta': net.RoBERTa_MLM.state_dict(),
                    'generate_model': net.generate_model.state_dict(),
                    'transformer': net.transformer_layer.state_dict(),
                    # 'transformer1':net.transformer_layer1.state_dict(),
                    # 'transformer2': net.transformer_layer2.state_dict(),
                    # 'transformer3': net.transformer_layer3.state_dict(),
                    # 'transformer4': net.transformer_layer4.state_dict()
                },
                'weights/best_model.pth'
            )

    # report
    print('Dev Acc={:.4f}, Dev F1_score={:.4f}'.format(acc / args.dev_size, f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                                     average='macro')), file=file_out)
    ############################################################################
    ##################################  test  ##################################
    ############################################################################
net = RoBERTa_MLM(args).cuda()
# net.load_state_dict(torch.load('weights/best_model.pth'))
best_mode = torch.load('weights/best_model.pth')
net.RoBERTa_MLM.load_state_dict(best_mode['roberta'])
# net.generate_model.load_state_dict(best_mode['generate_model'])
# net.transformer_layer.load_state_dict(best_mode['transformer'])
# net.transformer_layer1.load_state_dict(best_mode['transformer1'])
# net.transformer_layer2.load_state_dict(best_mode['transformer2'])
# net.transformer_layer3.load_state_dict(best_mode['transformer3'])
# net.transformer_layer4.load_state_dict(best_mode['transformer4'])
all_indices = torch.randperm(args.test_size).split(args.batch_size)
loss_epoch = []
acc = 0.0
f1_pred = torch.IntTensor([]).cuda()
f1_truth = torch.IntTensor([]).cuda()
net.eval()

# Just for Multi-Prompt case
'''
test_pred = torch.zeros(1474, 4)
test_truth = torch.zeros(1474, 4)
'''
with torch.no_grad():
    for batch_indices in all_indices:
        # get a batch of wordvecs
        batch_arg, mask_arg, token_mask_indices ,batch_t5_mask= get_batch_test(test_arg_1, test_arg_2, batch_indices)

        batch_arg = batch_arg.cuda()
        mask_arg = mask_arg.cuda()
        batch_t5_mask=batch_t5_mask.cuda()

        y = Variable(label_te[batch_indices]).cuda()

        encoder_output=net.RoBERTa_MLM.roberta(batch_arg, mask_arg)[0].cuda()
        # encoder_output1=net.transformer_layer(encoder_output)
        out_arg=net.RoBERTa_MLM.lm_head(encoder_output)

        out_vocab = torch.zeros(len(batch_arg), 50265).cuda()  # [16,50265]
        for i in range(len(batch_arg)):
            out_vocab[i] = out_arg[i][token_mask_indices[i]]  # [arg_len, vocab]
        out_ans = out_vocab[:, Token_id]  # Tensor.cuda()  [12,27]
        #
        # # Verbalizer
        pred_word = torch.argmax(out_ans, dim=1).tolist()  # list  12
        pred = torch.IntTensor(len(batch_arg), 4).cuda()  # [12,4]
        for tid, idx in enumerate(pred_word, 0):
            if idx <= 3:
                pred[tid] = torch.IntTensor([1, 0, 0, 0])
            elif 3 < idx <= 9:
                pred[tid] = torch.IntTensor([0, 1, 0, 0])
            elif 9 < idx <= 20:
                pred[tid] = torch.IntTensor([0, 0, 1, 0])
            elif 20 < idx <= 26:
                pred[tid] = torch.IntTensor([0, 0, 0, 1])
        out_sense=pred

        _, pred = torch.max(out_sense, dim=1)
        _, truth = torch.max(y, dim=1)
        num_correct = (pred == truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)
    # report
    print('Test Acc={:.4f}, Test F1_score={:.4f}'.format(acc / args.test_size, f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                                    average='macro')), file=file_out)
    print('Test Acc={:.4f}, Test F1_score={:.4f}'.format(acc / args.test_size, f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                                    average='macro')))

file_out.close()
