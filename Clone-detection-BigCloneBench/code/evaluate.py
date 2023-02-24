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
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512,pool=None):
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
        if 'test' not in postfix:
            data=random.sample(data,int(len(data)*0.1))

        self.examples=pool.map(get_example,tqdm(data,total=len(data)))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item].input_ids),torch.tensor(self.examples[item].label)

def load_and_cache_examples(tokenizer, 
                            test_data_file, 
                            block_size, 
                            evaluate=False,
                            test=False,
                            pool=None):
    dataset = TextDataset(tokenizer, file_path=test_data_file,block_size=block_size,pool=pool)
    return dataset

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

if __name__ == "__main__":
    
    checkpoints = ["/home/fjiriges/attention_bias/attentionBias/Clone-detection-BigCloneBench/code/saved_models/checkpoint-best-f1/model.bin",
                   "/home/fjiriges/attention_bias/attentionBias/Clone-detection-BigCloneBench/code/original_models/checkpoint-best-f1/model.bin"]
    
    names = ["attention_guided","original"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']

    config = config_class.from_pretrained('microsoft/codebert-base')

    tokenizer = tokenizer_class.from_pretrained('roberta-base')
    
    test_data_file = "../dataset/test.txt"
    block_size = 400
    cpu_cont = 16
    pool = multiprocessing.Pool(cpu_cont)
    eval_dataset = load_and_cache_examples(tokenizer,
                                        test_data_file,
                                        block_size,
                                        evaluate=True,pool=pool)
    
    eval_sampler = SequentialSampler(eval_dataset)
    eval_batch_size=32
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size,num_workers=4,pin_memory=True)
    
    for i in range(len(checkpoints)):
        model = model_class.from_pretrained('microsoft/codebert-base',
                                            config=config)
        model=Model(model,config,tokenizer)

        print("Load model from checkpoint", names[i])
        model.load_state_dict(torch.load(checkpoints[i]))
        model = model.to(device)

        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()
        logits=[]  
        y_trues=[]

        for batch in eval_dataloader:
            inputs = batch[0].to(device)        
            labels=batch[1].to(device) 
            with torch.no_grad():
                lm_loss,logit, a = model(block_size,inputs,labels)
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
            "eval_threshold":best_threshold}
        print(result)