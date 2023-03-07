import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import  RobertaConfig, RobertaModel, RobertaTokenizer
import argparse
import json
import os
from model2 import Model
import random
import multiprocessing
from tqdm import tqdm, trange
import numpy as np
import javalang
from tree_sitter import Language, Parser
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
        
        data=data[:1000]

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
                            evaluate=False,
                            test=False,
                            pool=None):
    dataset = TextDataset(tokenizer, file_path=test_data_file,block_size=block_size,pool=pool)
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

def traverse(code, node,depth=0):
    declaration = {}
    stack = []
    stack.append(node)
    while stack:
        node = stack.pop()
        if ('declaration' in node.type and node.type != "local_variable_declaration") or 'if_statement' in node.type or 'else' in node.type:
            data = code[node.start_byte:node.end_byte].split('{')[0].strip().split(' ')
            if node.type in declaration:
                declaration[node.type].append(data)
            else:
                declaration[node.type] = [data]
        for child in node.children:
            stack.append(child)
    return declaration

def label_tokens(token_list, declaration):
    types = [] 
    for token in token_list:
        flag = False
        for key in declaration:
            for value in declaration[key]:
                if token in value:
                    types.append(key)
                    flag = True
                    break
            if flag:
                break
        if not flag:
            types.append("other")
    return types

def get_extended_types(token_list, types):
    tree = list(javalang.tokenizer.tokenize(" ".join(token_list)))
    code = ' '.join(token_list)
    right = 0
    left = 0
    postion_mapping = [] 

    while right < len(code):
        if code[right] == ' ':
            postion_mapping.append((left, right))
            left = right + 1
        right += 1

    # add the last token
    postion_mapping.append((left, right))
    code = ["<s>"]
    extended_types = []
    left = 0
    for node in tree:
        # rewrite code
        node = str(node).split(' ')
        if node[1] == '"MASK"':
            code.append('<mask>')
        else:
            code.append(node[1][1:-1])
        # extend types
        left = int(node[-1]) -1
        right = left + len(node[1][1:-1])
        # check (left, right) in postion_mapping and get the index
        for i in range(len(postion_mapping)):
            if left >= postion_mapping[i][0] and right <= postion_mapping[i][1]:
                extended_types.append([types[i], node[1]])
                break
    code.append("</s>")
    return extended_types, ' '.join(code)


def get_ast_types(code):
    code = code.replace("{", " {")
    code = " ".join(code.split())
    code_list = code.split(' ')
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node
    declaration = traverse(code, root_node)
    types = label_tokens(code_list, declaration)

    ast_types, rewrote_code = get_extended_types(code_list, types)
    # check the index of first second value is the "{"
    if ast_types[0][1] == '"class"':
        return ['[CLS]'] + [i[0] for i in ast_types] + ['[SEP]'], rewrote_code
    index_ = 0
    # if not class declaration, find the first "{" and add method_declaration before it
    for i in range(len(ast_types)):
        if ast_types[i][1] == '"{"':
            index_ = i
            break
    final_types = [] 
    final_types.append('[CLS]')
    for i in range(len(ast_types)):
        if i < index_:
            final_types.append("method_declaration")
        else:
            final_types.append(ast_types[i][0])
    final_types.append('[SEP]')
    return np.array(final_types), rewrote_code

def get_start_end_of_token_when_tokenized(code_list, types, tokenizer):
  reindexed_types = []
  start = 0
  end = 0
  for each_token in code_list: 
      tokenized_list = tokenizer.tokenize(each_token)
      end += len(tokenized_list)
      reindexed_types.append((start, end-1))
      start = end
  return reindexed_types

def getSyntaxAttentionScore(model, data, tokenizer, syntaxList, model_type='finetuned'):
    block_size = 400
    all_instances = []
    number = 0
    data = data[:2000]
    for code_sample in tqdm(data):
        Instantce_Result = {}
        for syntaxType in syntaxList:
            Instantce_Result[syntaxType+model_type] = []
                    
        types_1, rewrote_code_1 = get_ast_types(code_sample[3])
        types_2, rewrote_code_2 = get_ast_types(code_sample[4])
        
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
        start_end = get_start_end_of_token_when_tokenized(rewrote_code_1.split(' '), types_1, tokenizer)
        
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
    Language.build_library(
	# Store the library in the `build` directory
	'build/my-languages.so',
	
	# Include one or more languages
	[
		'/Users/jirigesi/Documents/tree-sitter-java'
	]
    )

    JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
    parser = Parser()

    parser.set_language(JAVA_LANGUAGE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = RobertaConfig.from_pretrained('microsoft/codebert-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('microsoft/codebert-base',
                                        output_attentions=True, 
                                        output_hidden_states=True)

    model=Model(model,config,tokenizer)
    checkpoint_prefix = "/Users/jirigesi/Documents/icse2023/attentionBias/Clone-detection-BigCloneBench/code/saved_models/checkpoint-best-f1/model.bin"
    model.load_state_dict(torch.load(checkpoint_prefix, map_location='cpu'))
    model = model.to(device)
    
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
        lines = 1000
        added_lines = 0
        for line in f:
            if added_lines >= lines:
                break
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

    syntax_list = ['else', 
                    'if_statement', 
                    'method_declaration', 
                    'class_declaration', 
                    'constructor_declaration']
    
    # syntax_attention_weights = getSyntaxAttentionScore(model, data, tokenizer, syntax_list, model_type='finetuned')
    
    # # pikle the syntax_attention_weights
    # with open('ast_attention_weights_finetuned3.pkl', 'wb') as f:
    #     pickle.dump(syntax_attention_weights, f)
    model = RobertaModel.from_pretrained('microsoft/codebert-base',
                                        output_attentions=True, 
                                        output_hidden_states=True)
    model=Model(model,config,tokenizer)
    model = model.to(device)
    syntax_attention_weights = getSyntaxAttentionScore(model, data, tokenizer, syntax_list, model_type='pretrained')
    
    # pikle the syntax_attention_weights
    with open('ast_attention_weights_pretrained3.pkl', 'wb') as f:
        pickle.dump(syntax_attention_weights, f)