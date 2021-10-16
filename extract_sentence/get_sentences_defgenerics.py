import os
import json
import random 
from collections import defaultdict
import numpy as np


"""extract fdefinition sentence from wikitionary for each word"""

if __name__ == '__main__':

    parser = ArgumentParser() 
    parser.add_argument('-w','--words-list', help='list of all words that need sentences', required=True)
    parser.add_argument('-c','--corpus-file', help='corpus file', required=True)
    parser.add_argument('-s','--source', help='choose generics or wikitionary', choices=['generics', 'wikitionary'],required=True)
    parser.add_argument('-b','--build-folder', help='location of output files',required=True)

    args = parser.parse_args()

	if not os.path.exists(args.build_folder):
		os.makedirs(args.build_folder)

    print('loading words')
    words_set=set()
    for line in open(args.words_list,'r'):
        word=line.strip()
        if '/' in word:
            word = word.replace('/','_slash_')
        words_set.add(word)
    print('Loaded ',len(words_set),' words')


    if args.source == 'generics':
		""" The sentences from this datasets are originally from the following dataset: 'ARC', 'ConceptNet', 'SimpleWikipedia', 'TupleKB', 'Waterloo', 'WordNet3.0'"""
		GenericsKB = defaultdict(lambda:defaultdict())
		GenericsKB_all_sent = {}
		with open(args.corpus_file) as inf:
		    first_line = inf.readline()
		    for line in inf.readlines():
		        cols = line.strip().split('\t')
		        source = cols[0]
		        word = cols[1]
		        sent = cols[3]
		        score = cols[4]
		        if word in words_set:
		            if source in ['ARC','SimpleWikipedia','Waterloo']: # only these resource have natural sentences 
		                GenericsKB[word][sent]=score
		        else:
		            pass  
		# rank the sentences for each word according to the confidence score
		for word in GenericsKB.keys():
		    ls = [k for k,v in sorted(GenericsKB[word].items(), key = lambda item:item[1], reverse=True)]
		    GenericsKB_all_sent[word]=ls

		# pick top 20 sentences for each word
		GenericsKB_sentence = {}
		for word in GenericsKB_sent:
		    if len(GenericsKB_sent[word])>=20:
		        GenericsKB_sentence[word]=random.sample(GenericsKB_sent[word][:20],20)
		    else:
		        num = len(GenericsKB_sent[word])
		        GenericsKB_sentence[word]= random.sample(GenericsKB_sent[word],num)

		output_name = os.path.join(args.build_folder,'generics_sent.json')
		json.dump(GenericsKB_sentence,open(output_name,'w'))


	elif args.source == 'wikitionary':
		wiki_dictionary = {}
		with open(args.corpus_file) as inf:
		    for line in inf:
		        word_defin = line.strip()
		        word,defin = line.split('\t')
		        if word in words_set:
		            defin_new = word+' is '+defin
		            wiki_dictionary[word]=defin_new

		output_name = os.path.join(args.build_folder,'wiki_def_sent.json')
		json.dump(wiki_dictionary,open(output_name,'w'))