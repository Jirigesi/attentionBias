import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import  RobertaConfig, RobertaModel, RobertaTokenizer
import argparse
import json
import os
from model2 import Model
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import random
import multiprocessing
from tqdm import tqdm, trange
import numpy as np
import javalang
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
np.random.seed(0)
import seaborn as sns
import collections
import pickle
import sklearn
from matplotlib import cm
from sklearn import manifold
from sklearn.metrics import recall_score, precision_score, f1_score

class TextDataset(Dataset):
    def __init__(self, tokenizer, sample_size=1000, file_path='train', block_size=512,pool=None):
        postfix=file_path.split('/')[-1].split('.txt')[0]
        self.examples = []
        index_filename=file_path
        print("Creating features from index file at %s ", index_filename)
        url_to_code={}
        with open('/'.join(index_filename.split('/')[:-1])+'/data.jsonl') as f:
            for line in f:
                line=line.strip()
                js=json.loads(line)
                url_to_code[js['idx']]=js['func']
        data=[]
        cache={}
        f=open(index_filename)
        with open(index_filename) as f:
            for line in f:
                line=line.strip()
                url1,url2,label=line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                if label=='0':
                    label=0
                else:
                    label=1
                data.append((url1,url2,label,tokenizer,cache,url_to_code))
        data=data[:sample_size]

        self.examples=pool.map(get_example,tqdm(data,total=len(data)))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item].input_ids),torch.tensor(self.examples[item].label)

def get_example(item):
    url1,url2,label,tokenizer,cache,url_to_code=item
    if url1 in cache:
        code1=cache[url1].copy()
    else:
        try:
            code=' '.join(url_to_code[url1].split())
        except:
            code=""
        code1=tokenizer.tokenize(code)
    if url2 in cache:
        code2=cache[url2].copy()
    else:
        try:
            code=' '.join(url_to_code[url2].split())
        except:
            code=""
        code2=tokenizer.tokenize(code)
        
    return convert_examples_to_features(code1,code2,label,url1,url2,tokenizer,block_size, cache)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 url1,
                 url2):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.url1=url1
        self.url2=url2
        
def load_and_cache_examples(tokenizer, 
                            test_data_file,
                            block_size, 
                            sample_size=1000,
                            evaluate=False,
                            test=False,
                            pool=None):
    dataset = TextDataset(tokenizer, 
                          file_path=test_data_file,
                          sample_size=sample_size,
                          block_size=block_size,
                          pool=pool)
    return dataset

def convert_examples_to_features(code1_tokens,code2_tokens,label,url1,url2,tokenizer,block_size,cache):
    code1_tokens=code1_tokens[:block_size-2]
    code1_tokens =[tokenizer.cls_token]+code1_tokens+[tokenizer.sep_token]
    code2_tokens=code2_tokens[:block_size-2]
    code2_tokens =[tokenizer.cls_token]+code2_tokens+[tokenizer.sep_token]  
    
    code1_ids=tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = block_size - len(code1_ids)
    code1_ids+=[tokenizer.pad_token_id]*padding_length
    
    code2_ids=tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = block_size - len(code2_ids)
    code2_ids+=[tokenizer.pad_token_id]*padding_length
    
    source_tokens=code1_tokens+code2_tokens
    source_ids=code1_ids+code2_ids
    return InputFeatures(source_tokens,source_ids,label,url1,url2)

def get_syntax_types_for_code(code_snippet):
    types = ["[CLS]"]
    code = ["<s>"]
    tree = list(javalang.tokenizer.tokenize(code_snippet))
    
    for i in tree:
        j = str(i)
        j = j.split(" ")
        if j[1] == '"MASK"':
            types.append('[MASK]')
            code.append('<mask>')
        else:
            types.append(j[0].lower())
            code.append(j[1][1:-1])
        
    types.append("[SEP]")
    code.append("</s>")
    return np.array(types), ' '.join(code)

def get_start_end_of_token_when_tokenized(code, types, tokenizer):
    reindexed_types = []
    start = 0
    end = 0
    for index, each_token in enumerate(code.split(" ")):
        tokenized_list = tokenizer.tokenize(each_token)
        for i in range(len(tokenized_list)):
            end += 1
        reindexed_types.append((start, end-1))
        start = end
    return reindexed_types

def getSyntaxAttentionScore(model, data, tokenizer, syntaxList, model_type='finetuned'):
    block_size = 400
    all_instances = []
    number = 0
    for code_sample in tqdm(data):
        Instantce_Result = {}
        for syntaxType in syntaxList:
            Instantce_Result[syntaxType+model_type] = []

        types_1, rewrote_code_1 = get_syntax_types_for_code(code_sample[3])
        types_2, rewrote_code_2 = get_syntax_types_for_code(code_sample[4])
        
        tokenized_ids_1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(rewrote_code_1))
        tokenized_ids_2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(rewrote_code_2))

        if len(tokenized_ids_2) > 400:
            tokenized_ids_2 = tokenized_ids_2[:399] + [tokenizer.sep_token_id]

        if len(tokenized_ids_1) > 400:
            tokenized_ids_1 = tokenized_ids_1[:399] + [tokenizer.sep_token_id]
        
        padding_length = block_size - len(tokenized_ids_1)
        tokenized_ids_1+=[tokenizer.pad_token_id]*padding_length
        padding_length = block_size - len(tokenized_ids_2)
        tokenized_ids_2+=[tokenizer.pad_token_id]*padding_length

        source_ids = tokenized_ids_1 + tokenized_ids_2
        labels = code_sample[2]
        source_ids = torch.tensor(source_ids).unsqueeze(0).to(device)
        labels = torch.tensor(labels).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(block_size,source_ids,labels)

        _attention = output[2].attentions
        start_end = get_start_end_of_token_when_tokenized(rewrote_code_1, types_1, tokenizer)
        
        for syntaxType in syntaxList:
            attention_weights = [[[] for col in range(12)] for row in range(12)]
            for layer in range(12):
                for head in range(12):
                    for each_sep_index in np.where(types_1==syntaxType)[0]:
                        start_index, end_index = start_end[each_sep_index]
                        interim_value = _attention[layer][0][head][:, start_index:end_index+1].mean().cpu().detach().numpy()
                        if np.isnan(interim_value):
                            pass
                        else: 
                            attention_weights[layer][head].append(interim_value)     
            if np.array(attention_weights).shape[2] != 0:
                Instantce_Result[syntaxType+model_type].append(np.array(attention_weights))
                            
        all_instances.append(Instantce_Result)
        
    return all_instances






if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = RobertaConfig.from_pretrained('microsoft/codebert-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('microsoft/codebert-base',
                                        output_attentions=True, 
                                        output_hidden_states=True)

    model=Model(model,config,tokenizer)
    checkpoint_prefix = "saved_models/checkpoint-best-f1/model.bin"
    model.load_state_dict(torch.load(checkpoint_prefix))
    model = model.to(device)
    print("model loaded")
    
    file_path = "../dataset/valid.txt"
    postfix=file_path.split('/')[-1].split('.txt')[0]
    index_filename=file_path
    url_to_code={}
    with open('/'.join(index_filename.split('/')[:-1])+'/data.jsonl') as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            url_to_code[js['idx']]=js['func']
        data=[]
        cache={}
        f=open(index_filename)
        with open(index_filename) as f:
            added_lines = 0
            for line in f:
                line=line.strip()
                url1,url2,label=line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                if label=='0':
                    label=0
                else:
                    label=1
                data.append((url1,url2,label,' '.join(url_to_code[url1].split()), ' '.join(url_to_code[url2].split())))
                added_lines += 1
                
                
    syntax_list = ['annotation', 'basictype', 'boolean', 
          'decimalinteger', 'identifier', 'keyword',
          'modifier', 'operator', 'separator', 'null',
          'string', 'decimalfloatingpoint']
    
    test_data_file = "../dataset/valid.txt"
    block_size = 400
    cpu_cont = 16
    pool = multiprocessing.Pool(cpu_cont)
    eval_dataset = load_and_cache_examples(tokenizer,
                                        test_data_file,
                                        block_size,
                                        sample_size=5000,
                                        evaluate=True,pool=pool)
    
    eval_sampler = SequentialSampler(eval_dataset)
    eval_batch_size=32
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size,num_workers=4,pin_memory=True)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]

    for batch in eval_dataloader:
        inputs = batch[0].to(device)        
        labels=batch[1].to(device) 
        with torch.no_grad():
            lm_loss,logit,a = model(block_size,inputs,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    
    best_threshold=0
    best_f1=0
    for i in range(1,100):
        threshold=i/100
        y_preds=logits[:,1]>threshold
        recall=recall_score(y_trues, y_preds)
        precision=precision_score(y_trues, y_preds)
        f1=f1_score(y_trues, y_preds) 
        if f1>best_f1:
            best_f1=f1
            best_threshold=threshold
            
    y_preds=logits[:,1]>best_threshold
    recall=recall_score(y_trues, y_preds)
    precision=precision_score(y_trues, y_preds)   
    f1=f1_score(y_trues, y_preds)             
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold":best_threshold,
        
    }
    print("Summary: ", result)

    # get index for mispredcited and correctly predicted 
    correct_precition_index = []
    misprediction_index = []

    for index, each in enumerate(y_preds):
        if each == y_trues[index]:
            correct_precition_index.append(index)
        else:
            misprediction_index.append(index)
            
    correct_codes = []
    mispredic_codes = []
    for i in correct_precition_index:
        correct_codes.append(data[i])
        
    for i in misprediction_index:
        mispredic_codes.append(data[i])
        
    syntax_attention_correct = getSyntaxAttentionScore(model,
                                                   correct_codes,
                                                   tokenizer,
                                                   syntax_list,
                                                   model_type='finetuned')
    syntax_attention_mis = getSyntaxAttentionScore(model,
                                               mispredic_codes,
                                               tokenizer,
                                               syntax_list,
                                               model_type='finetuned')
    
    # pickle the result 
    with open('syntax_attention_correct.pkl', 'wb') as f:
        pickle.dump(syntax_attention_correct, f)
    print('syntax_attention_correct.pkl saved')
        
    with open('syntax_attention_mis.pkl', 'wb') as f:
        pickle.dump(syntax_attention_mis, f)
        
    print('syntax_attention_mis.pkl saved')
    