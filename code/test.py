import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import f1_score,confusion_matrix
from load_data import load_data
from prepro_data import prepro_data_train, prepro_data_dev, prepro_data_test
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup,T5Tokenizer
from parameter import parse_args
from tqdm import tqdm
from prepro_data import generate_prompt
from model import RoBERTa_MLM
import time
args = parse_args()
net = RoBERTa_MLM(args).cuda()
model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
tokenizer1=T5Tokenizer.from_pretrained('./t5-base')
# f=open('test_result_explanation_w_prompt1.txt','a',encoding='utf-8')
best_mode=torch.load('weights/best_model.pth')
net.RoBERTa_MLM.load_state_dict(best_mode['roberta'])
net.transformer_layer.load_state_dict(best_mode['transformer'])
# net.transformer_layer1.load_state_dict(best_mode['transformer1'])
# net.transformer_layer2.load_state_dict(best_mode['transformer2'])
# net.transformer_layer3.load_state_dict(best_mode['transformer3'])
# net.transformer_layer4.load_state_dict(best_mode['transformer4'])
net.generate_model.load_state_dict(best_mode['generate_model'])
dict_answer={0:'Comparison', 1:'Contingency', 2:'Expansion', 3:'Temporal'}
all_indices = torch.arange(args.test_size).split(args.batch_size)
loss_epoch = []
acc = 0.0
f1_pred = torch.IntTensor([]).cuda()
f1_truth = torch.IntTensor([]).cuda()
net.eval()
train_data, dev_data, test_data= load_data()
test_arg_1, test_arg_2, test_label,test_label_conn= prepro_data_test(test_data)

label_te = torch.LongTensor(test_label)
Token_id=[1712,21507,53,959,
           142,25,98,28766,4634,187,
           1386,1195,50,8,67,43681,4327,1246,78,5329,4010,
           172,8960,1433,656,71,6637]

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
        encode_dict_t5=tokenizer1.encode_plus(
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

with torch.no_grad():
    N=0
    start = time.time()
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
        label=[]
    # # Verbalizer
        pred_word = torch.argmax(out_ans, dim=1).tolist()  # list  12
        pred = torch.IntTensor(len(batch_arg), 4).cuda()  # [12,4]
        for tid, idx in enumerate(pred_word, 0):
            if idx <= 3:
                pred[tid] = torch.IntTensor([1, 0, 0, 0])
                label.append('comparison')
            elif 3 < idx <= 9:
                pred[tid] = torch.IntTensor([0, 1, 0, 0])
                label.append('contingency')
            elif 9 < idx <= 20:
                pred[tid] = torch.IntTensor([0, 0, 1, 0])
                label.append('expansion')
            elif 20 < idx <= 26:
                pred[tid] = torch.IntTensor([0, 0, 0, 1])
                label.append('temporal')
        out_sense=pred
        _, pred = torch.max(out_sense, dim=1)
        _, truth = torch.max(y, dim=1)

        arg1 = batch_arg.clone()
        mask_arg1 = mask_arg.clone()
        new_arg, new_mask_arg = generate_prompt(arg1, mask_arg1, label, tokenizer)
        #
        encoder_output = net.RoBERTa_MLM.roberta(new_arg, new_mask_arg)[0].cuda()
        encoder_output1 = net.transformer_layer(encoder_output)
        # encoder_output2 = net.transformer_layer1(encoder_output1)
        # encoder_output3 = net.transformer_layer2(encoder_output2)
        # encoder_output4 = net.transformer_layer3(encoder_output3)
        # encoder_output5 = net.transformer_layer4(encoder_output4)
        # for j in range(len(batch_arg)):
        #     decoder_input_ids = torch.tensor([[37]]).cuda()
        #     for _ in range(250):
        #         decoder_output = net.generate_model.decoder(decoder_input_ids,
        #                                                     encoder_hidden_states=encoder_output1[j].unsqueeze(
        #                                                         0),
        #                                                     encoder_attention_mask=batch_t5_mask[j].unsqueeze(
        #                                                         0))
        #         logits = net.generate_model.lm_head(decoder_output.last_hidden_state)
        #         next_token_logits = logits[:, -1, :]  # [1,32128]
        #         next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(0)  # [1,1]
        #         decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)
        #     generated_text = tokenizer1.decode(decoder_input_ids[0], skip_special_tokens=True)
        #     pred_answer=dict_answer[pred[j].item()]
        #     truth_answer=dict_answer[truth[j].item()]
        #     # print('原文本： ', tokenizer1.decode(batch_ids_t5[j], skip_special_tokens=True))
        #     f.write(truth_answer+'||'+pred_answer+'||'+generated_text+'\n')
            # print('生成结果:', generated_text)
            # print(pred_answer)
            # print(generated_text)
            # N = N + 1
            # if N == 50:
            #     end = time.time()
            #     print("50句的时间为：", end - start)


        _, pred = torch.max(out_sense, dim=1)
        _, truth = torch.max(y, dim=1)
        num_correct = (pred == truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)

    # report
    print('Test Acc={:.4f}, Test F1_score={:.4f}'.format(acc / args.test_size, f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                                    average='macro')))
    print(f1_score(f1_truth.cpu(),f1_pred.cpu(),average=None))
    # f.close()
