from argparse import ArgumentParser
import os
import sys
from collections import defaultdict
from datetime import datetime
import multiprocessing as mp
from nltk.tokenize import sent_tokenize,word_tokenize
import json
import pickle
import random

def ctx_length(corpus_path):
    word_length_count=defaultdict(lambda:defaultdict(int))
    ### count lines
    print('Counting corpus lines')
    linecount=0
    for line in open(corpus_path,'r'):
        linecount+=1
    print('This file has ',linecount,' lines')

    ### extract sentence
    lc=0
    for line in open(corpus_path,'r'):
        line=line.strip()
        if not line.startswith('<doc id') and not line.startswith('*****__'):
            sents = sent_tokenize(line)
            for sent in sents:
                tokens = word_tokenize(sent)
                token_set = set(tokens)
                length = len(tokens)
                if 6<=length<=64:
                    if 6<=length<15:
                        outf_dir = folder_10
                    elif 15<=length<25:
                        outf_dir = folder_20
                    elif 25<=length<35:
                        outf_dir = folder_30
                    elif 35<=length<45:
                        outf_dir = folder_40
                    elif 45<=length<=64:
                        outf_dir = folder_55
                    for w in token_set:
                        if w in words_set:
                            if word_length_count[w][outf_dir]<20:
                                outf_name = os.path.join(outf_dir,w)
                                with open(outf_name,'a') as outfile:
                                    outfile.write(sent+'\n')
                                word_length_count[w][outf_dir]+=1
        lc+=1
        if lc % 1000 == 0:
            print('Done ',lc,' lines of ',linecount,' of file ',corpus_path,' | At time: ',datetime.now())



def ctx(corpus_path):
    word_count = defaultdict(int)
    ### count lines
    print('Counting corpus lines')
    linecount=0
    for line in open(corpus_path,'r'):
        linecount+=1
    print('This file has ',linecount,' lines')
    outf_dir = folder_all
    lc=0
    ### extract sentence
    for line in open(corpus_path,'r'):
        line=line.strip()
        if not line.startswith('<doc id') and not line.startswith('*****__'):
            sents = sent_tokenize(line)
            for line in open(corpus_path,'r'):
                line=line.strip()
                if not line.startswith('<doc id') and not line.startswith('*****__'):
                    sents = sent_tokenize(line)
                    for sent in sents:
                        tokens = word_tokenize(sent)
                        token_set = set(tokens)
                        length = len(tokens)
                        if 6<=length<=64:
                            for w in token_set:
                                if w in words_set:
                                    if word_count[w]<20:
                                        outf_name = os.path.join(outf_dir,w)
                                        with open(outf_name,'a') as outfile:
                                            outfile.write(sent+'\n')
                                        word_count[w]+=1
                lc+=1
                if lc % 1000 == 0:
                    print('Done ',lc,' lines of ',linecount,' of file ',corpus_path,' | At time: ',datetime.now())

def file2dic(file_ls):
    D = defaultdict(list)
    for f in file_ls:
        word = f.split('/')[-1][:-4]
        with open(f,'r') as inf:
            for line in inf:
                line = line.strip()
                D[word].append(line)
    return D


if __name__ == '__main__':

    parser = ArgumentParser() 
    parser.add_argument('-w','--words-file', help='words file', required=True)
    parser.add_argument('-c','--corpus-file', help='corpus file', required=True) 
    parser.add_argument('-t','--intermiate-folder', help='folder that store intermiate files ', required=True)
    parser.add_argument('-b','--build-folder', help='folder that store the output files', required=True)
    parser.add_argument('-l','--truncate', help='retrict the token length of sentences', choices=['false', 'true'],required=True)

    args = parser.parse_args()

    
    # split the wikipedia corpus into several smaller files

    print('Counting lines in original corpus')
    linecount=0
    with open(args.corpus_file) as f:
      for line in f:
          linecount+=1
    workers=mp.cpu_count()
    sents_per_split=round(linecount/workers)
    print('Source corpus has ',linecount,' lines')

    print('Splitting original corpus in ',workers,' files of ~',sents_per_split,' lines')
    if not os.path.exists(args.intermiate_folder):
      os.makedirs(args.intermiate_folder)

    linecount=0
    splitcount=0
    outf=open(os.path.join(args.intermiate_folder,'split_'+str(splitcount))+'.txt','w')
    with open(args.corpus_file,'r') as f:
      for line in f:
          linecount+=1
          outf.write(line)
          if linecount % sents_per_split == 0:
              outf.close()
              splitcount+=1
              outf=open(os.path.join(args.intermiate_folder,'split_'+str(splitcount)+'.txt'),'w')
              print('Saved split numb: ',splitcount,' of ',workers)


    # loading vocabulary 

    print('loading words')
    words_set=set()
    for line in open(args.words_file,'r'):
        word=line.strip()
        if '/' in word:
            word = word.replace('/','_slash_')
        words_set.add(word)
    print('Loaded ',len(words_set),' words')

    # collect all splitted wikipedia text file
    splits=[os.path.join(args.intermiate_folder,inf) for inf in os.listdir(args.intermiate_folder) 
    if inf.startswith('split') 
    and inf.endswith('.txt') 
    and not 'triples' in inf]

    print('Processing files:')
    for i in splits:
        print(i)

    # extract random sentences
    if not os.path.exists(args.build_folder):
        os.makedirs(args.build_folder)

    p = mp.Pool(processes=workers)

    if args.truncate == 'true':
        folder_10 = os.path.join(args.intermiate_folder,'10')
        folder_20 = os.path.join(args.intermiate_folder,'20')
        folder_30 = os.path.join(args.intermiate_folder,'30')
        folder_40 = os.path.join(args.intermiate_folder,'40')
        folder_55 = os.path.join(args.intermiate_folder,'55')
        os.makedirs(folder_10)
        os.makedirs(folder_20)
        os.makedirs(folder_30)
        os.makedirs(folder_40)
        os.makedirs(folder_55)
        p.map(ctx_length,splits)
        p.close()

        ls_10=[os.path.join(folder_10,f) for f in os.listdir(folder_10)
        ls_20=[os.path.join(folder_20,f) for f in os.listdir(folder_20)
        ls_30=[os.path.join(folder_30,f) for f in os.listdir(folder_30)
        ls_40=[os.path.join(folder_40,f) for f in os.listdir(folder_40)
        ls_55=[os.path.join(folder_55,f) for f in os.listdir(folder_55)

        all_ls = [ls_10,ls_20,ls_30,ls_40,ls_55]
        all_name = ['wiki_random_10.json','wiki_random_20.json','wiki_random_30.json','wiki_random_40.json','wiki_random_55.json']

        for i in range(len(all_ls)):
            ls = all_ls[i]
            file_name = os.path.join(args.build_folder,all_name[i])
            word2sent_selected = {}
            word2sent = file2dic(ls)
            for t,s in word2sent.items():
                num = min(len(s),20)
                word2sent_selected[t]=random.sample(s,num)
            json.dump(word2sent_selected,open(file_name,w))
       

    if args.truncate == 'false':
        folder_all = os.path.join(args.intermiate_folder,'all')
        os.makedirs(folder_all)
        p.map(ctx,splits)
        p.close()

        file_ls = [os.path.join(folder_all,f) for f in os.listdir(folder_all)]
        file_name = os.path.join(args.build_folder,'wiki_random_all.json')

        word2sent_selected = {}
        word2sent = file2dic(file_ls)
        for t,s in word2sent.items():
            num = min(len(s),20)
            word2sent_selected[t]=random.sample(s,num)
        json.dump(word2sent_selected,open(file_name,w))

    print('Done')

    

