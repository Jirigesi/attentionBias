import torch
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
import argparse
import json
import os
import pandas as pd
import numpy as np 
from tqdm import tqdm
from scipy import stats
import javalang
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients, LayerActivation
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import pickle

if torch.__version__ >= '1.7.0':
    norm_fn = torch.linalg.norm
else:
    norm_fn = torch.norm
    
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

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / norm_fn(attributions)
    return attributions

def construct_whole_bert_embeddings(input_ids, ref_input_ids):
    input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids)
    ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids)
    
    return input_embeddings, ref_input_embeddings

def predict_forward_func(input_embeddings, tokenized_text):
    output = model(inputs_embeds=input_embeddings)
    if tokenizer.mask_token_id not in tokenized_text:
        print("Mask token not in tokenized text")
    index = tokenized_text.index(tokenizer.mask_token_id)
    # print("[MASK] Index: {}, length of tokenized text is {}".format(index, len(tokenized_text)))
    if index > output.logits.shape[1]:
        print("Length of output is {} and index is {}".format(output.logits.shape[1], index))
    output_list = output.logits[0][index]
    output_list = output_list.unsqueeze(0)
    
    return output_list.max(1).values

def get_CLS_attention(model, tokenizer, codes):
    # layer x head x code
    cls_data = [[[] for col in range(12)] for row in range(12)]
    with torch.no_grad():
        failed = 0
        for code in tqdm(codes):
            try:
                
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code))
                input_ids = torch.tensor([tokenized_text]).to(device)
                reference_indices = token_reference.generate_reference(input_ids.shape[1], device=device).unsqueeze(0)

                layer_attrs = []
                layer_attn_mat = []
                input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(input_ids, reference_indices)

                for i in range(model.config.num_hidden_layers):
                    lc = LayerConductance(predict_forward_func, 
                                        model.roberta.encoder.layer[i])
                    layer_attributions = lc.attribute(inputs=input_embeddings, 
                                                            baselines=ref_input_embeddings, 
                                                            additional_forward_args=(tokenized_text))
                    layer_attrs.append(summarize_attributions(layer_attributions[0]))
                    layer_attn_mat.append(layer_attributions[1])
                # layer x seq_len
                layer_attrs = torch.stack(layer_attrs)
                # layer x batch x head x seq_len x seq_len
                layer_attn_mat = torch.stack(layer_attn_mat)
                for layer in range(12):
                    for head in range(12):
                        cls_data[layer][head].append(layer_attn_mat[layer][0][head][:, 0:1].mean().cpu().detach().numpy())
            except:
                failed += 1
    print("Failed: {}".format(failed))
    
    return cls_data


def get_SEP_attention(model, tokenizer, codes):
    sep_data = [[[] for col in range(12)] for row in range(12)]

    with torch.no_grad():
        failed = 0
        try:
            for code in tqdm(codes):
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code))
                input_ids = torch.tensor([tokenized_text]).to(device)
                reference_indices = token_reference.generate_reference(input_ids.shape[1], device=device).unsqueeze(0)

                layer_attrs = []
                layer_attn_mat = []
                input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(input_ids, reference_indices)

                for i in range(model.config.num_hidden_layers):
                    lc = LayerConductance(predict_forward_func, 
                                        model.roberta.encoder.layer[i])
                    layer_attributions = lc.attribute(inputs=input_embeddings, 
                                                            baselines=ref_input_embeddings, 
                                                            additional_forward_args=(tokenized_text))
                    layer_attrs.append(summarize_attributions(layer_attributions[0]))
                    layer_attn_mat.append(layer_attributions[1])
                # layer x seq_len
                layer_attrs = torch.stack(layer_attrs)
                # layer x batch x head x seq_len x seq_len
                layer_attn_mat = torch.stack(layer_attn_mat)
                for layer in range(12):
                    for head in range(12):
                        for each_sep_index in torch.where(input_ids[0]==2)[0].cpu().detach().numpy():
                            sep_data[layer][head].append(layer_attn_mat[layer][0][head][:, each_sep_index].mean().cpu().detach().numpy()) 
        except:
            failed += 1
    print("Failed: {}".format(failed))
    return sep_data

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

def getInstanceSyntaxAttributionScore(codes, tokenizer, syntaxList):
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
                # get reference indices
                reference_indices = token_reference.generate_reference(input_ids.shape[1], device=device).unsqueeze(0)
                # get layer attribution
                layer_attrs = []
                layer_attn_mat = []
                input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(input_ids, reference_indices)
                for i in range(model.config.num_hidden_layers):
                    lc = LayerConductance(predict_forward_func, 
                                        model.roberta.encoder.layer[i])
                    layer_attributions = lc.attribute(inputs=input_embeddings, 
                                                            baselines=ref_input_embeddings, 
                                                            additional_forward_args=(tokenized_text))
                    layer_attrs.append(summarize_attributions(layer_attributions[0]))
                    layer_attn_mat.append(layer_attributions[1])
                # layer x seq_len
                layer_attrs = torch.stack(layer_attrs)
                # layer x batch x head x seq_len x seq_len
                layer_attn_mat = torch.stack(layer_attn_mat)
                # get start and end index of each token
                start_end = get_start_end_of_token_when_tokenized(rewrote_code, types, tokenizer)
                
                for syntaxType in syntaxList:
                    attribution_scores = [[[] for col in range(12)] for row in range(12)]
                    for layer in range(12):
                        for head in range(12):
                            for each_sep_index in np.where(types==syntaxType)[0]:
                                start_index, end_index = start_end[each_sep_index]
                                interim_value = layer_attn_mat[layer][0][head][:, start_index:end_index+1].mean().cpu().detach().numpy()
                                if np.isnan(interim_value):
                                    pass
                                else: 
                                    attribution_scores[layer][head].append(interim_value)
                    
                    if np.array(attribution_scores).shape[2] != 0:
                        Instantce_Result[syntaxType].append(np.array(attribution_scores))

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
    
    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence

    token_reference = TokenReferenceBase(reference_token_idx=ref_token_id)
    
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

    # number_of_samples = 30 # a small set of data for testing
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
            predict_word_id = torch.argmax(score_list).cpu().data.tolist()
        
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
        
    interpretable_embedding = configure_interpretable_embedding_layer(model, 'roberta.embeddings.word_embeddings')
        
    #############################
    ## CLS attribution
    #############################
    # print('Calculating CLS attribution....')
    # CLS_atten_correct = get_CLS_attention(model, 
    #                                       tokenizer, 
    #                                       correct_codes)
    # # CLS_atten_sum_correct = np.sum(CLS_atten_correct, axis=1)
    
    # CLS_atten_misprediction = get_CLS_attention(model, 
    #                                             tokenizer, 
    #                                             mispredic_codes)
    # # CLS_atten_sum_misprediction = np.sum(CLS_atten_misprediction, axis=1)
    
    # # pickle the CLS_atten_sum into the results folder
    # CLS_atten_sum_correct_file = 'results/CLS_attri_sum_correct.pkl'
    # with open(CLS_atten_sum_correct_file, 'wb') as f:
    #     pickle.dump(CLS_atten_correct, f)
        
    # CLS_atten_sum_misprediction_file = 'results/CLS_attri_sum_misprediction.pkl'
    # with open(CLS_atten_sum_misprediction_file, 'wb') as f:
    #     pickle.dump(CLS_atten_misprediction, f)
        
    #############################
    ## SEP attention
    #############################
    # print('Calculating SEP attention weights....')
    # SEP_atten_correct = get_SEP_attention(model, 
    #                                       tokenizer, 
    #                                       correct_codes)
    # # SEP_atten_sum_correct = np.sum(SEP_atten_correct, axis=1)
    
    # SEP_atten_misprediction = get_SEP_attention(model,
    #                                             tokenizer,
    #                                             mispredic_codes)
    # SEP_atten_sum_misprediction = np.sum(SEP_atten_misprediction, axis=1)
       
    # # pickle the SEP_atten_sum into the results folder
    # SEP_atten_sum_correct_file = 'results/SEP_attri_sum_correct.pkl'
    # with open(SEP_atten_sum_correct_file, 'wb') as f:
    #     pickle.dump(SEP_atten_correct, f)
        
    # SEP_atten_sum_misprediction_file = 'results/SEP_attri_sum_misprediction.pkl'
    # with open(SEP_atten_sum_misprediction_file, 'wb') as f:
    #     pickle.dump(SEP_atten_misprediction, f)
        
    #############################
    ## Syntax attention
    #############################
    print('Calculating Program Syntax attention weights....')
    

    syntax_list = ['annotation', 'basictype', 'boolean', 
            'decimalinteger', 'identifier', 'keyword',
            'modifier', 'operator', 'separator', 'null',
            'string', 'decimalfloatingpoint']
        
    

    correct_attributions = getInstanceSyntaxAttributionScore(correct_codes, 
                                                                tokenizer, 
                                                                syntax_list)
        
    
    mispredict_attributions = getInstanceSyntaxAttributionScore(mispredic_codes, 
                                                            tokenizer, 
                                                            syntax_list)

    # pickle the syntax attention into the results folder
    syntax_attri_sum_correct_file = 'results/syntax_attribution_instance_correct6.pkl'
    with open(syntax_attri_sum_correct_file, 'wb') as f:
        pickle.dump(correct_attributions, f)
        
    syntax_attri_sum_misprediction_file = 'results/syntax_attribution_instances_misprediction6.pkl'
    with open(syntax_attri_sum_misprediction_file, 'wb') as f:
        pickle.dump(mispredict_attributions, f)