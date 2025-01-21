# -*- coding: utf-8 -*-

import re
import torch
import numpy as np

"""
conn_label = ['by being', 'on the one hand', 'in response', 'particularly', 'on the whole', 'as of now', 'while', 'although', 'while being', 'still',
               'next', 'specifically', 'if we are', 'if one is to', 'thereby', 'ultimately', 'as a consequence', 'because it was', 'nonetheless', 'with',
               'with the purpose of', 'for one thing', 'despite this', 'accordingly', 'if she is', 'all the while', 'namely', 'eventually', 'more specifically', 'given',
               'separately', 'in the end', 'granted', 'and', 'as a result of being', 'to this end', 'in general', 'though', 'consequently', 'in summary',
               'likewise', 'in sum', 'if one were', 'whereas', 'when', 'it is because', 'with the goal', 'if I am', 'at the time', 'in order to',
               'because of being', 'in being', 'were one to', 'during that time', 'in order for them', 'if you are', 'thus being', 'insofar as', 'now', 'through',
               'for the reason that', 'for that purpose', 'as', 'for example', 'by means of', 'on account of being', 'in comparison', 'further', 'since then', 'thus',
               'since', 'simultaneously', 'furthermore', 'as a result of having', 'conversely', 'on the other hand', 'instead', 'in particular', 'in comparison to the fact', 'as part of that',
               'if one is', 'alternatively', 'besides', 'after', 'soon', 'considering that', 'at the same time', 'nevertheless', "what's more", 'in other words',
               'also', 'first', 'additionally', 'before', 'that is', 'as it turns out', 'but', 'in short', 'even though', 'similarly',
               'with the goal of', 'as a result of', 'in addition', 'in', 'after being', 'in the meantime', 'rather', 'yet', 'thereafter', 'in this case',
               'if we were', 'before that', 'for instance', 'upon', 'inasmuch as', 'however', 'afterwards', 'despite', 'in contrast', 'by contrast',
               'This is because', 'later', 'if it was', 'in addition to being', 'overall', 'given that', 'for', 'therefore', 'after all', 'so that',
               'despite being', 'previously', 'in order to be', 'then', 'otherwise', 'finally', 'or', 'indeed', 'in fact', 'prior to this',
               'if they were', 'more to the point', 'if they are', 'in addition to', 'for the purpose of', 'in order', 'plus', 'moreover', 'as such', 'by comparison',
               'if it is', 'at that time', 'by', 'if', 'for that reason', 'because of that', 'so as', 'if he is', 'after having', 'meanwhile',
               'subsequently', 'earlier', 'on the contrary', 'as a result', 'hence', 'regardless', 'in more detail', 'because', 'incidentally', 'as evidence',
               'generally', 'so', 'if it were', 'because of']
"""

class_label = ['Comparison', 'Contingency', 'Expansion', 'Temporal']

subtype_label = ['Comparison.Similarity', 'Comparison.Contrast', 'Comparison.Concession+SpeechAct.Arg2-as-denier+SpeechAct', 'Comparison.Concession.Arg2-as-denier', 'Comparison.Concession.Arg1-as-denier', 
                 'Contingency.Purpose.Arg2-as-goal', 'Contingency.Purpose.Arg1-as-goal', 'Contingency.Condition+SpeechAct', 'Contingency.Condition.Arg2-as-cond', 'Contingency.Condition.Arg1-as-cond', 'Contingency.Cause+SpeechAct.Result+SpeechAct', 'Contingency.Cause+SpeechAct.Reason+SpeechAct', 'Contingency.Cause+Belief.Result+Belief', 'Contingency.Cause+Belief.Reason+Belief', 'Contingency.Cause.Result', 'Contingency.Cause.Reason', 
                 'Expansion.Substitution.Arg2-as-subst', 'Expansion.Manner.Arg2-as-manner', 'Expansion.Manner.Arg1-as-manner', 'Expansion.Level-of-detail.Arg2-as-detail', 'Expansion.Level-of-detail.Arg1-as-detail', 'Expansion.Instantiation.Arg2-as-instance', 'Expansion.Instantiation.Arg1-as-instance', 'Expansion.Exception.Arg2-as-excpt', 'Expansion.Exception.Arg1-as-excpt', 'Expansion.Equivalence', 'Expansion.Disjunction', 'Expansion.Conjunction', 
                 'Temporal.Synchronous', 'Temporal.Asynchronous.Succession', 'Temporal.Asynchronous.Precedence']

subtype_label_word = ['similarly', 'but', 'but', 'however', 'although',
                      'for', 'for', 'if', 'if', 'if', 'because', 'so', 'because', 'so', 'because', 'so', 
                      'instead', 'by', 'thereby', 'specifically', 'specifically', 'specifically', 'specifically', 'and', 'and', 'and', 'and', 'and',
                      'simultaneously', 'previously', 'then']


ans_word={'although':0,'nevertheless':1,'but':2,'however':3,'because':4,'as':5,'so':6,'consequently':7,'thus':8,'since':9,
          'instead':10,'rather':11,'or':12,'and':13,'also':14,'furthermore':15,'instance':16,'example':17,'first':18,
          'indeed':19,'specifically':20,'then':21,'subsequently':22,'previously':23,'earlier':24,'after':25,
          'meanwhile':26}


def prepro_data_train(train_file_list):

    train_label = []
    train_arg_1 = []
    train_arg_2 = []
    train_label_list = []
    train_label_conn=[]
    train_explanation=[]
    train_prompt=[]

    for i in range(len(train_file_list)):
        sentence = train_file_list[i].split('||')
        train_arg_1.append(sentence[1])
        train_arg_2.append(sentence[2])
        train_label.append(eval(sentence[0])[0])
        train_label_conn.append(ans_word[eval(sentence[0])[2]])
        train_explanation.append(sentence[3])
        # train_prompt.append(train_expand[i])

    for cla in train_label:
        if cla == class_label[0]:
            train_label_list.append([1, 0, 0, 0])
        elif cla == class_label[1]:
            train_label_list.append([0, 1, 0, 0])
        elif cla == class_label[2]:
            train_label_list.append([0, 0, 1, 0])
        elif cla == class_label[3]:
            train_label_list.append([0, 0, 0, 1])
        else:
            print('error')

    return train_arg_1, train_arg_2, train_label_list,train_label_conn,train_explanation

def prepro_data_dev(dev_file_list):

    dev_label = []
    dev_arg_1 = []
    dev_arg_2 = []
    dev_label_list = []
    dev_label_conn=[]
    dev_explanation=[]
    dev_prompt=[]

    for i in range(len(dev_file_list)):
        sentence = dev_file_list[i].split('||')
        dev_arg_1.append(sentence[1])
        dev_arg_2.append(sentence[2])
        dev_label.append(eval(sentence[0])[0])
        dev_label_conn.append(ans_word[eval(sentence[0])[2]])
        dev_explanation.append(sentence[3])
        # dev_prompt.append(dev_expand[i])

    for cla in dev_label:
        if cla == class_label[0]:
            dev_label_list.append([1, 0, 0, 0])
        elif cla == class_label[1]:
            dev_label_list.append([0, 1, 0, 0])
        elif cla == class_label[2]:
            dev_label_list.append([0, 0, 1, 0])
        elif cla == class_label[3]:
            dev_label_list.append([0, 0, 0, 1])
        else:
            print('error')

    return dev_arg_1, dev_arg_2, dev_label_list,dev_label_conn,dev_explanation

def prepro_data_test(test_file_list):

    test_label = []
    test_arg_1 = []
    test_arg_2 = []
    test_label_list = []
    test_label_conn=[]
    test_prompt=[]

    for i in range(len(test_file_list)):
        sentence = test_file_list[i].split('|||')
        test_arg_1.append(sentence[1])
        test_arg_2.append(sentence[2])
        test_label.append(eval(sentence[0])[0])
        test_label_conn.append(ans_word[eval(sentence[0])[2]])
        # test_prompt.append(test_expand[i])

    for cla in test_label:
        if cla == class_label[0]:
            test_label_list.append([1, 0, 0, 0])
        elif cla == class_label[1]:
            test_label_list.append([0, 1, 0, 0])
        elif cla == class_label[2]:
            test_label_list.append([0, 0, 1, 0])
        elif cla == class_label[3]:
            test_label_list.append([0, 0, 0, 1])
        else:
            print('error')

    return test_arg_1, test_arg_2, test_label_list,test_label_conn

def generator(generate_model,trans_layer,encoder,batch_mask_t5):
    text=[]
    encoder_output=trans_layer(encoder)
    for j in range(len(encoder_output)):
        decoder_input_ids = torch.tensor([[37]]).cuda()
        for _ in range(200):
            decoder_output = generate_model.decoder(decoder_input_ids,
                                                        encoder_hidden_states=encoder_output[j].unsqueeze(
                                                            0),
                                                        encoder_attention_mask=batch_mask_t5[j].unsqueeze(
                                                            0))
            logits = generate_model.lm_head(decoder_output.last_hidden_state)
            next_token_logits = logits[:, -1, :]  # [1,32128]
            next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(0)  # [1,1]
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)
        # generated_text = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
        text.append(decoder_input_ids[0])
    return text

def generate_prompt(arg,mask,label,tokenizer):
    for i in range(len(arg)):
        position=np.argwhere(np.array(arg[i].cpu()) == 50264)[0][0]
        s=label[i]+', the reason is that '
        ids_add=tokenizer.encode(s)
        for j in range(len(ids_add)-1):
            arg[i][position]=ids_add[j+1]
            mask[i][position]=1
            position+=1
    return arg,mask