import os
from transformers import BertTokenizer, BertForMaskedLM, BertModel, RobertaTokenizer, RobertaForMaskedLM, RobertaModel
import torch
from torch.utils.data import SequentialSampler, TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import logging


def init_logging_path(log_path,task_name,file_name):
    dir_log = os.path.join(log_path,f"{task_name}/{file_name}/")
    if os.path.exists(dir_log) and os.listdir(dir_log):
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
             os.utime(dir_log, None)
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
             os.utime(dir_log, None)
    return dir_log

def getMaskTokenizer(version):
    tokenizer = ""
    cls_token = ""
    sep_token = ""
    mask_token = ""
    if version.split('-')[0]=='bert':
        tokenizer = BertTokenizer.from_pretrained(version)
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        mask_token = "[MASK]"
    elif version.split('-')[0] == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(version)
        cls_token = "<s>"
        sep_token = "</s>"
        mask_token = "<mask>"
    return tokenizer, cls_token, sep_token, mask_token

def getUnmaskTokenizer(version):
    tokenizer = ""
    cls_token = ""
    sep_token = ""
    if version.split('-')[0] == 'bert':
        tokenizer = BertTokenizer.from_pretrained(version)
        cls_token = "[CLS]"
        sep_token = "[SEP]"
    elif version.split('-')[0] == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(version)
        cls_token = "<s>"
        sep_token = "</s>"
    return tokenizer, cls_token, sep_token

def tokenize_mask(data_zip, max_seq_len, tokenizer, cls_token, sep_token, mask_token):
    token_ids = []
    input_mask = []
    indices = []
    for d in data_zip:
        sent, word = d
        #sent = line.strip().lower()
        left_seq, _, right_seq = sent.partition(str(word))
        tokens = [cls_token] + tokenizer.tokenize(left_seq)
        idx = len(tokens)
        tokens += [mask_token]
        tokens += tokenizer.tokenize(right_seq) + [sep_token]
        # print(tokens)
        if len(tokens) > max_seq_len: continue
        indices.append(idx)
        t_id = tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (max_seq_len - len(t_id))
        i_mask = [1] * len(t_id) + padding
        t_id += padding
        token_ids.append(t_id)
        input_mask.append(i_mask)
    return token_ids, input_mask, indices


def tokenize_nomask(data_zip, max_seq_len,  tokenizer, cls_token, sep_token):
    token_ids = []
    indices = []
    input_mask = []
    for d in data_zip:
            sent, word = d
            #sent = line.strip().lower()
            left_seq, _, right_seq = sent.partition(str(word))
            tokens = [cls_token] + tokenizer.tokenize(left_seq)
            word_tokens = tokenizer.tokenize(word)
            idx = [len(tokens), len(word_tokens)]
            tokens += word_tokens
            tokens += tokenizer.tokenize(right_seq) + [sep_token]

            # print(tokens)
            if len(tokens) > max_seq_len:
                continue
            indices.append(idx)
            t_id = tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (max_seq_len - len(t_id))
            i_mask = [1] * len(t_id) + padding
            t_id += padding
            token_ids.append(t_id)
            input_mask.append(i_mask)
    return token_ids, input_mask, indices

def getMaskedLmModel(version):
    mask_model = ""    
    if version.split('-')[0] == 'bert':
        mask_model = BertForMaskedLM.from_pretrained(version, output_hidden_states=True)
    elif version.split('-')[0] == 'roberta':
        mask_model = RobertaForMaskedLM.from_pretrained(version, output_hidden_states=True)
    return mask_model



def extract_mv_mask(token_ids, input_mask, word_indices, batch_size, mask_model, usegpu=False):
    tokens_tensor = torch.tensor([ids for ids in token_ids], dtype=torch.long)
    input_mask_tensor = torch.tensor([im for im in input_mask], dtype=torch.long)
    indices = torch.tensor(word_indices)
    data = TensorDataset(tokens_tensor, input_mask_tensor, indices)

    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if usegpu:
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            logging.info("use multiple gpu, n_gpu = " + str(n_gpu))
            mask_model = torch.nn.DataParallel(mask_model)
        mask_model.to('cuda')  # use GPU

    num_layers = mask_model.module.config.num_hidden_layers if isinstance(mask_model,
                                                                     torch.nn.DataParallel) else mask_model.config.num_hidden_layers
    hidden_dim = mask_model.module.config.hidden_size if isinstance(mask_model,
                                                               torch.nn.DataParallel) else mask_model.config.hidden_size
    results = torch.empty(num_layers, len(word_indices), hidden_dim)

    # if usegpu:
    #     results = results.to('cuda')
    start_idx = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        token_id, in_mask, index = batch
        if usegpu:
            token_id = token_id.to('cuda')  # GPU
            in_mask = in_mask.to('cuda')
            # index = index.to('cuda')

        with torch.no_grad():
            hidden_states = mask_model(token_id, attention_mask=in_mask)[1]  # all hidden-states(initial_embedding + all hidden layers), batch_size, sequence_length, dim

        # print('hidden states: ', len(hidden_states), hidden_states[0].shape)
        sent_ids = list(range(hidden_states[0].shape[0]))
        end_idx = start_idx + len(sent_ids)
        for i in range(1, len(hidden_states)):  # 0-layer is the initial embedding
            results[i-1][start_idx:end_idx] = hidden_states[i][sent_ids, index]
            torch.cuda.empty_cache()
        start_idx = end_idx
    return results


def getModel(version):
    model = ""
    if version.split('-')[0] == 'bert':
        model = BertModel.from_pretrained(version, output_hidden_states=True)#.get_input_embeddings()
    elif version.split('-')[0] == 'roberta':
        model = RobertaModel.from_pretrained(version, output_hidden_states=True)
    return model 



def extract_mv_nomask(token_ids, input_mask, word_indices, batch_size, model, usegpu=False):

    tokens_tensor = torch.tensor([ids for ids in token_ids], dtype=torch.long)
    input_mask_tensor = torch.tensor([im for im in input_mask], dtype=torch.long)
    indices = torch.tensor([id for id in word_indices], dtype=torch.int)
    data = TensorDataset(tokens_tensor, input_mask_tensor, indices)

    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if usegpu:
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model.to('cuda')  # use GPU

    num_layers = model.module.config.num_hidden_layers if isinstance(model, torch.nn.DataParallel) else model.config.num_hidden_layers
    hidden_dim = model.module.config.hidden_size if isinstance(model, torch.nn.DataParallel) else model.config.hidden_size
    results = torch.empty(num_layers, len(word_indices), hidden_dim)
    #if usegpu:
    #    results = results.to('cuda')
    start_idx = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        token_id, in_mask, index = batch
        if usegpu:
            token_id = token_id.to('cuda')  # GPU
            in_mask = in_mask.to('cuda')

        with torch.no_grad():
            hidden_states = model(token_id, attention_mask=in_mask)[2]  # all hidden states, 0 is initial embedding
        # print('hidden states: ', len(hidden_states), hidden_states[0].shape)
        end_idx = start_idx + hidden_states[0].shape[0]
        s_idx = index[:, 0]
        num = index[:, 1]
        for l in range(1, len(hidden_states)):
            avg_vec = torch.tensor([], dtype=torch.float)
            if usegpu:
                avg_vec = avg_vec.to('cuda')
            for ii in range(hidden_states[l].shape[0]):
                vec = hidden_states[l][ii, s_idx[ii]: s_idx[ii] + num[ii]]
                avg_vec = torch.cat((avg_vec, torch.mean(vec, dim=0).view(1, -1)))

            results[l-1][start_idx:end_idx] = avg_vec.cpu()
            torch.cuda.empty_cache()
        start_idx = end_idx
    return results

def read_sentences(sent_file):
    sents = []
    with open(sent_file,'r') as inf:
        first_line = inf.readline().strip()
        #print(first_line)
        topic = first_line.split(' ')[-1]
        #print(topic)
        word = ' '.join(first_line.split(' ')[:-1])
        #print(word)
        line_count = 0
        for line in inf.readlines():
            if line_count < 100:
                    line_count += 1
                    line = line.strip().split('___',1)[-1]
                    #line = line.strip().split('___')[1]
                    #print(line)
                    if not line.startswith('Section::::'):
                        sents.append(line)     
            else: 
              break            
    return sents, word, topic 

def read_sentences_2(sent_file):
    sents = []
    with open(sent_file,'r') as inf:
        first_line = inf.readline().strip()
        #print(first_line)
        topic = first_line.split(' ')[-1]
        #print(topic)
        word = topic
        #print(word)
        line_count = 0
        for line in inf.readlines():
            if line_count < 100:
                    line_count += 1
                    #line = line.strip().split('___',1)[-1]
                    #line = line.strip().split('___')[1]
                    #print(line)
                    if not line.startswith('Section::::'):
                        sents.append(line)     
            else: 
              break            
    return sents, word, topic 

