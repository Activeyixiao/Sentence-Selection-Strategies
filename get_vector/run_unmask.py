import logging
import os
import sys
import getopt
import csv
import json
from collections import defaultdict
import torch
import numpy as np
import pickle

project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
from utils import *

def main(argv):
    log_dir = os.path.join(os.getcwd(), "log/")
    logfile = ''

    max_seq_length = 128
    batch_size = 64
    input_dir = ''  # path of sentence files
    output_dir = ''
    bert_version = ''
    usegpu = False

    try:
        opts, args = getopt.getopt(argv, "hl:i:s:b:o:v:g:", ["lfile=", "idir=", "max_seq_length=", "batch_size=", "odir=", "bert_version=", "usegpu="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('run_script.py  -l <logfile> -i <input_dir> -s <max_seq_length> -b <batch_size> -o <output_dir> -v <bert_version> -g')
            sys.exit()
        elif opt in ("-l", "--lfile"):
            logfile = arg
        elif opt in ("-i", "--idir"):  # path of sentence files
            input_dir = arg
        elif opt in ("-s", "--max_seq_length"):
            batch_size = int(arg)    
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-o", "--odir"):  # path to store mention vector files
            output_dir = arg
        elif opt in ("-v", "--bert_version"):
            bert_version = arg
        elif opt in ("-g", "--usegpu"):
            if arg.lower() in ("true", "yes", "y"):
                usegpu = True



    log_file_path = init_logging_path(log_dir,"unmask_vectors", logfile)
    logger = logging.getLogger('server_logger')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("start logging: " + log_file_path)
    logger.info('input dir is ' + input_dir)
    logger.info('output dir is ' + output_dir)
    logger.info('BERT version is ' + bert_version)
    logger.info('max Seq length ' + str(max_seq_length))
    logger.info('batch_size is ' + str(batch_size))
    logger.info('GPU ' + str(usegpu))
    

    output_dir_last = output_dir+'/'+'last_layer'
    output_dir_avg = output_dir+'/'+'avg_layers'

    if not os.path.exists(output_dir_last):
        os.makedirs(output_dir_last)
    if not os.path.exists(output_dir_avg):
        os.makedirs(output_dir_avg)

    tokenizer, cls_token, sep_token = getUnmaskTokenizer(bert_version)
    model = getModel(bert_version)

    
    word_sents = json.load(open(input_dir,'r'))
    for word in word_sents.keys():
        try :
            logger.info("*****************************************")
            logger.info(word)
            sents=word_sents[word]
            words = [word] * len(sents)
            token_ids, input_mask, indices = tokenize_nomask(zip(sents, words), max_seq_length,tokenizer, cls_token, sep_token)
            if len(token_ids) < 1:
                logging.info(word + " has no sentence")
                continue
            results =  extract_mv_nomask(token_ids, input_mask, indices, batch_size, model,  usegpu=usegpu) 
            last_layer = results[-1] # last hidden layer
            avg_layer = torch.mean(results, dim=0).cpu()  # average among all hidden layers, shape = (20 sents * 1024 dim)
            torch.save(last_layer, os.path.join(output_dir_last,word+'.pt'))
            torch.save(avg_layer, os.path.join(output_dir_avg,word+'.pt'))
        except: 
            print('file exeption: ' + word)
            logger.info('file exeption:' + word)              
    print('Done')
    logger.info('Done')
if __name__ == '__main__':
    main(sys.argv[1:])
