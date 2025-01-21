# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForMaskedLM, RobertaTokenizer,T5ForConditionalGeneration,T5Tokenizer
from prepro_data import generate_prompt

class RoBERTa_MLM(nn.Module):
    def __init__(self, args):
        super(RoBERTa_MLM, self).__init__()

        self.RoBERTa_MLM = RobertaForMaskedLM.from_pretrained('roberta-base')
        for param in self.RoBERTa_MLM.parameters():
            param.requires_grad = True
        
        self.vocab_size = args.vocab_size
        self.num_class = args.num_class
        self.dropout=nn.Dropout(args.dropout)
        self.generate_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1)
        # self.transformer_layer1 = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1)
        # self.transformer_layer2 = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1)
        # self.transformer_layer3 = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1)
        # self.transformer_layer4 = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.tokenizer=RobertaTokenizer.from_pretrained('roberta-base')
        # for param in self.transformer_layer.parameters():
        #     param.requires_grad=False
        # for param in self.generate_model.parameters():
        #     param.requires_grad=False
        self.f1=nn.Linear(768,4)

        #输入的ids，输入的attention_mask，mask在输入中的位置，三级连接词对应的ids，转折类中三级连接词数量，因果类中三级连接词数量，扩展类中三级连接词数量，顺承类中三级连接词数量
    def forward(self, arg, mask_arg, token_mask_indices, Token_id,batch_mask_t5,batch_expla_ids):

        # out_arg = self.RoBERTa_MLM(arg, mask_arg)[0].cuda()  # [batch, arg_len, vocab] [16,100,50265]对应的logits
        encoder_output=self.RoBERTa_MLM.roberta(arg,mask_arg)[0].cuda()
        # encoder1=self.RoBERTa_MLM.roberta(batch_prompt_ids,batch_prompt_mask)[0].cuda()
        encoder_output=self.dropout(encoder_output)
        out_arg=self.RoBERTa_MLM.lm_head(encoder_output)

        out_vocab = torch.zeros(len(arg), self.vocab_size).cuda() #[16,50265]
        for i in range(len(arg)):
            out_vocab[i] = out_arg[i][token_mask_indices[i]]  # [arg_len, vocab]
        
        out_ans = out_vocab[:, Token_id] # Tensor.cuda()  [12,27]

        label=[]
        label_logit=[]
        # Verbalizer
        pred_word = torch.argmax(out_ans, dim=1).tolist() # list  12
        pred = torch.IntTensor(len(arg), self.num_class).cuda() #[12,4]
        for tid, idx in enumerate(pred_word, 0):
            if idx <=3:
                pred[tid] = torch.IntTensor([1, 0, 0, 0])
                label.append('comparison')
            elif 3<idx<=9 :
                pred[tid] = torch.IntTensor([0, 1, 0, 0])
                label.append('contingency')
            elif 9<idx<=20:
                pred[tid] = torch.IntTensor([0, 0, 1, 0])
                label.append('expansion')
            elif 20<idx<=26:
                pred[tid] = torch.IntTensor([0, 0, 0, 1])
                label.append('temporal')

        arg1 = arg.clone()
        mask_arg1 = mask_arg.clone()
        new_arg, new_mask_arg = generate_prompt(arg1, mask_arg1, label, self.tokenizer)
        new_encoder = self.RoBERTa_MLM.roberta(new_arg, new_mask_arg)[0].cuda()

        encoder_output1 = self.transformer_layer(encoder_output)
        # encoder_output2=self.transformer_layer1(encoder_output1)
        # encoder_output3 = self.transformer_layer2(encoder_output2)
        # encoder_output4 = self.transformer_layer3(encoder_output3)
        # encoder_output5 = self.transformer_layer4(encoder_output4)
        # encoder_output1 = self.dropout(encoder_output1)

        #
        expla_ids = batch_expla_ids[:, :-1]
        decoder_output = self.generate_model.decoder(input_ids=expla_ids, encoder_hidden_states=encoder_output1,
                                                     encoder_attention_mask=batch_mask_t5)
        logits = self.generate_model.lm_head(decoder_output.last_hidden_state)

        return pred, out_ans, logits