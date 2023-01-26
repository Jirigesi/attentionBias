import torch
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
import argparse
import json
import os
import pandas as pd
import numpy as np 
from tqdm import tqdm
from scipy import stats
import pickle
import javalang

def get_cloze_words(filename, tokenizer):
    with open(filename, 'r', encoding='utf-8') as fp:
        words = fp.read().split('\n')
    idx2word = {tokenizer.encoder[w]: w for w in words}
    return idx2word

def read_answers(filename):
    answers = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            answers[line.split('<CODESPLIT>')[0]] = line.split('<CODESPLIT>')[1]
    return answers

def get_CLS_attention(model, tokenizer, codes):
    cls_data = np.zeros((12,12))
    with torch.no_grad():
        for eachCode in tqdm(codes, desc='CLS attention'):
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(eachCode))[:512]
            input_ids = torch.tensor([tokenized_text]).to(device)
            output_from_model = model(input_ids)
            
            _attention = output_from_model["attentions"]# attention shape is layers, batchsize, heads, tokenLen, tokenLen
            
            for layer in range(12):
                for head in range(12):
                    cls_data[layer][head] += _attention[layer][0][head][:, 0:1].mean().cpu().detach().numpy() # CLS attention

    CLS_atten = cls_data/len(codes)
    return CLS_atten

def get_SEP_attention(model, tokenizer, codes):
    sep_data = np.zeros((12,12))

    with torch.no_grad():
        for eachCode in tqdm(codes, desc='SEP attention'):
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(eachCode))
            # index = inputs.index(tokenizer.mask_token_id)
            inputs_id = torch.tensor([tokenized_text]).to(device)
            output_from_model = model(inputs_id)
            
            _attention = output_from_model["attentions"]# attention shape is layers, batchsize, heads, tokenLen, tokenLen
            
            for layer in range(12):
                for head in range(12):
                    for each_sep_index in torch.where(inputs_id[0]==tokenizer.sep_token_id)[0].cpu().detach().numpy():
                        sep_data[layer][head] += _attention[layer][0][head][:, each_sep_index].mean().cpu().detach().numpy() 

    SEP_atten = sep_data/len(codes)
    return SEP_atten

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

def getSyntaxAttentionScore(codes, tokenizer, syntaxList):
    # initialize the result dict with empty list for each syntax type
    all_instances = []
    with torch.no_grad():
        for eachCode in tqdm(codes):
            try: 
                Instantce_Result = {}
                for syntaxType in syntaxList:
                    Instantce_Result[syntaxType] = []
                    
                cleancode = eachCode.replace("<s> ", "").replace(" </s>", "").replace('<mask>', 'MASK')
                types, rewrote_code = get_syntax_types_for_code(cleancode)
                # send input to model
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(rewrote_code))
                input_ids = torch.tensor([tokenized_text]).to(device)
                output_from_model = model(input_ids)
                # get attention from model
                _attention = output_from_model["attentions"]# attention shape is layers, batchsize, heads, tokenLen, tokenLen
                # get start and end index of each token
                start_end = get_start_end_of_token_when_tokenized(rewrote_code, types, tokenizer)
                
                for syntaxType in syntaxList:
                    attention_weights = [[[] for col in range(12)] for row in range(12)]
                    for layer in range(12):
                        for head in range(12):
                            for each_sep_index in np.where(types==syntaxType)[0]:
                                start_index, end_index = start_end[each_sep_index]
                                interim_value = _attention[layer][0][head][:, start_index:end_index+1].mean().cpu().detach().numpy()
                                if np.isnan(interim_value):
                                    pass
                                else: 
                                    attention_weights[layer][head].append(interim_value)
                                    
                    if np.array(attention_weights).shape[2] != 0:
                        Instantce_Result[syntaxType].append(np.array(attention_weights))
                        
            except Exception as e:
                print(e)
            
            all_instances.append(Instantce_Result)
    return all_instances

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Runing on device:', device)
    MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)}

    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    config = config_class.from_pretrained('roberta-base')
    tokenizer = tokenizer_class.from_pretrained('roberta-base')

    model = RobertaForMaskedLM.from_pretrained('microsoft/codebert-base-mlm', 
                                            output_attentions=True, output_hidden_states=True)
    
    model.to(device)

    cloze_results = []
    cloze_words_file = 'data/cloze-all/cloze_test_words.txt'
    file_path = 'data/cloze-all/java/clozeTest.json'
    
    idx2word = get_cloze_words(cloze_words_file, tokenizer)
    
    lines = json.load(open(file_path))
    
    print('Total number of code instances: ', len(lines))
    
    answer_file = 'evaluator/answers/java/answers.txt'
    answers = read_answers(answer_file)
    answer_list = list(answers.values())
    
    bestSampleWithMaxPairLength = []
    bestSampleWithMaxPairLength_LEN =[]

    # number_of_samples = 100 # a small set of data for testing
    for i in range(len(lines)):
        code = ' '.join(lines[i]['pl_tokens'])
        bestStr = "<s> " + code + " </s>"
        bestLen = len(bestStr.split(" "))
        bestSampleWithMaxPairLength.append(bestStr)
        bestSampleWithMaxPairLength_LEN.append(bestLen)
        
    lengths=[]
    codes=[]
    selected_answers = []

    for index, code in enumerate(bestSampleWithMaxPairLength):
        l = len(tokenizer.tokenize(code))
        if l<=256:
            lengths.append(l)
            codes.append(code)
            selected_answers.append(answer_list[index])
            
    print('Selected number of codes: ', len(codes))
    print('Selected number of answers: ', len(selected_answers))
    
    # Split the codes into correct and incorrect preidted codes
    correct_precition_index = []
    misprediction_index = []
    for i, code in enumerate(codes): 
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code))
        index = tokenized_text.index(tokenizer.mask_token_id)
        input_ids = torch.tensor([tokenized_text]).to(device)

        with torch.no_grad():
            scores = model(input_ids)[0]
            score_list = scores[0][index]
            word_index = torch.LongTensor(list(idx2word.keys()))
            word_index = torch.zeros(score_list.shape[0]).scatter(0, word_index, 1).to(device)
            score_list = score_list + (1-word_index) * -1e6
            predict_word_id = torch.argmax(score_list).data.tolist()
        
        predict_word = idx2word[predict_word_id]
        if predict_word == selected_answers[i]:
            correct_precition_index.append(i)
        else:
            misprediction_index.append(i)
    print('Number of correct prediction: ', len(correct_precition_index))
    print('Number of misprediction: ', len(misprediction_index))
    
    correct_codes = []
    mispredic_codes = []
    for i in correct_precition_index:
        correct_codes.append(codes[i])
        
    for i in misprediction_index:
        mispredic_codes.append(codes[i])
    # #############################
    # ## CLS attention
    # #############################
    # print('Calculating CLS attention weights....')
    # CLS_atten_correct = get_CLS_attention(model, 
    #                                       tokenizer, 
    #                                       correct_codes)
    # CLS_atten_sum_correct = np.sum(CLS_atten_correct, axis=1)
    
    # CLS_atten_misprediction = get_CLS_attention(model, 
    #                                             tokenizer, 
    #                                             mispredic_codes)
    # CLS_atten_sum_misprediction = np.sum(CLS_atten_misprediction, axis=1)
    
    # # pickle the CLS_atten_sum into the results folder
    # CLS_atten_sum_correct_file = 'results/CLS_atten_sum_correct.pkl'
    # with open(CLS_atten_sum_correct_file, 'wb') as f:
    #     pickle.dump(CLS_atten_sum_correct, f)
        
    # CLS_atten_sum_misprediction_file = 'results/CLS_atten_sum_misprediction.pkl'
    # with open(CLS_atten_sum_misprediction_file, 'wb') as f:
    #     pickle.dump(CLS_atten_sum_misprediction, f)
    
    # #############################
    # ## SEP attention
    # #############################
    # print('Calculating SEP attention weights....')
    # SEP_atten_correct = get_SEP_attention(model, 
    #                                       tokenizer, 
    #                                       correct_codes)
    # SEP_atten_sum_correct = np.sum(SEP_atten_correct, axis=1)
    
    # SEP_atten_misprediction = get_SEP_attention(model,
    #                                             tokenizer,
    #                                             mispredic_codes)
    # SEP_atten_sum_misprediction = np.sum(SEP_atten_misprediction, axis=1)
       
    # # pickle the SEP_atten_sum into the results folder
    # SEP_atten_sum_correct_file = 'results/SEP_atten_sum_correct.pkl'
    # with open(SEP_atten_sum_correct_file, 'wb') as f:
    #     pickle.dump(SEP_atten_sum_correct, f)
        
    # SEP_atten_sum_misprediction_file = 'results/SEP_atten_sum_misprediction.pkl'
    # with open(SEP_atten_sum_misprediction_file, 'wb') as f:
    #     pickle.dump(SEP_atten_sum_misprediction, f)
        
    #############################
    ## Syntax attention
    #############################
    print('Calculating Program Syntax attention weights....')
    
    syntax_list = ['annotation', 'basictype', 'boolean', 
          'decimalinteger', 'identifier', 'keyword',
          'modifier', 'operator', 'separator', 'null',
          'string', 'decimalfloatingpoint']
    
    correct_attention_weights = getSyntaxAttentionScore(correct_codes, 
                                                        tokenizer, 
                                                        syntax_list)
        
    mispredict_attention_weights = getSyntaxAttentionScore(mispredic_codes, 
                                                            tokenizer, 
                                                            syntax_list)
    # pickle the syntax attention into the results folder
    syntax_atten_correct_file = 'results/syntax_attention_instance_correct5.pkl'
    with open(syntax_atten_correct_file, 'wb') as f:
        pickle.dump(correct_attention_weights, f)
        
    syntax_atten_sum_misprediction_file = 'results/syntax_attention_instance_misprediction5.pkl'
    with open(syntax_atten_sum_misprediction_file, 'wb') as f:
        pickle.dump(mispredict_attention_weights, f)

        
        
    