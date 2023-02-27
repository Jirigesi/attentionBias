# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class Model(nn.Module):
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
        self.softmax=nn.Softmax(dim=1)
    
    def forward(self, 
                input_ids=None,
                labels=None, 
                syntax_atten_matrix=None, 
                attn_head_types='0,1'): 
        # print("Input ids shape:")
        # print(input_ids.shape)
        input_ids=input_ids.view(-1,self.args.block_size)
        a = self.encoder(input_ids=input_ids, 
                         attention_mask=input_ids.ne(1))
        outputs = a[0]
        attention = a[-1]
        logits=self.classifier(outputs)
        prob=self.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            ag_loss = self.compute_ag_loss(attention, 
                                           syntax_atten_matrix,
                                           self.args.device, 
                                           attn_head_types=attn_head_types)
            loss = loss + ag_loss
            return loss,prob
        else:
            return prob
    
    def compute_ag_loss(self,
                        attentions, 
                        syntax_atten_matrix,
                        device, 
                        attn_head_types='0,1'):
        '''
        Adds a random loss based on attention values
        To test gradients
        outputs[-1] contains the attention values (tuple of size num_layers)
        and each elements is of the shape
        [batch_size X num_heads X max_sequence_len X max_sequence_len]
        '''
        # Get the attention head types
        attn_head_types = [int(i) for i in attn_head_types.split(',')]
        # The number attention heads of each type. one-to-one, next, previous, first
        numbers = attn_head_types
        cum_sum = np.cumsum(numbers)
        # Matrices containing the attention patterns
        print("attentions shape:")
        temp = []

        for attention in attentions:
            temp.append(attention[::2])
        attentions = temp
        print(len(attentions))
        print(attentions[0].shape)
        print("syntax_atten_matrix shape:")
        print(syntax_atten_matrix.shape)
        # one_to_one - pays attention to the corresponding token
        one_to_one = torch.eye(self.args.block_size)
        # next_token - pays attention to the next token
        # next_token = torch.cat((torch.cat((torch.zeros(attentions[0].shape[-1]-1, 1), torch.eye(attentions[0].shape[-1]-1)), dim=1),\
        #                         torch.zeros(1, attentions[0].shape[-1])), dim=0)
        # prev_token - pays attention to the previous token
        
        # prev_token = torch.cat((torch.zeros(1, attentions[0].shape[-1]), \
        #     torch.cat((torch.eye(attentions[0].shape[-1]-1), torch.zeros(attentions[0].shape[-1]-1, 1)), dim=1)), dim=0)
        # cls_token  - pays attention to the first index ([CLS])
        # cls_token = torch.zeros(attentions[0].shape[-1], attentions[0].shape[-1])
        # cls_token[:,0] = 1.

        targets = [one_to_one, syntax_atten_matrix]

        # Loss for positional attention patterns
        expanded_targets = []
        loss = torch.nn.MSELoss()
        total_loss = 0.        
        # Change the tensor's dimension
        for (num, target) in zip(numbers, targets):
            if num == 0:
                expanded_targets.append(None)
            else:
                # Add dimensions so that the tensor can be repeated
                target = torch.unsqueeze(target, 0)
                # Change the target tensor's dimension so that it matches batch_size X num_heads[chosen]
                target = target.repeat(attentions[0].shape[1], 1, 1, 1 )
                # change target from head*batch*seq*seq to batch*head*seq*seq
                target = target.permute(1,0,2,3)
                target = target.to(device)
                expanded_targets.append(target)

        # Go over all the layers
        for i in range(len(attentions)):
            for j in range(len(numbers)):
                if expanded_targets[j] is not None:
                    if j == 0:
                        total_loss += loss(expanded_targets[j], attentions[i][:,0:cum_sum[j]])
                    else:
                        total_loss += loss(expanded_targets[j], attentions[i][:,cum_sum[j-1]:cum_sum[j]])
        return total_loss
