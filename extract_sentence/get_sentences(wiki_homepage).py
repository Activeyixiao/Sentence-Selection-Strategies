import os
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import random
import shutil
import json
import pickle
import torch

def extract_wiki_sentences(wiki_loc):
    lemmatizer = WordNetLemmatizer()
    word_sents = defaultdict(list)
    for title,loc in wiki_loc.items():
        title_lem = lemmatizer.lemmatize(title)
        with open(loc,'r') as inf:
            for line in inf.readlines():
                line = line.strip()
                sents = sent_tokenize(line)
                for sent in sents:
                    word_ls = [lemmatizer.lemmatize(w) for w in word_tokenize(sent)]
                    if title in word_ls or title_lem in word_ls:
                        word_sents[title].append(sent)
    return word_sents

if __name__ == '__main__':

    parser = ArgumentParser() 
    parser.add_argument('-w','--words-list', help='list of words that need sentences', required=True)
    parser.add_argument('-c','--corpus-dir', help='corpus folder', required=True)
    parser.add_argument('-b','--build-folder', help='location of output files',required=True)

    args = parser.parse_args()

    if not os.path.exists(args.build_folder):
        os.makedirs(args.build_folder)


    # loading vocabulary 

    print('loading words')
    words_set=set()
    for line in open(args.words_list,'r'):
        word=line.strip()
        if '/' in word:
            word = word.replace('/','_slash_')
        words_set.add(word)
    print('Loaded ',len(words_set),' words')


	title_location = {}
	for d in [os.path.join(args.corpus_dir,i) for i in os.listdir(args.corpus_dir)]:
	    all_files = [os.path.join(d,f) for f in os.listdir(d) if not '/' in f]
	    for file in all_files:
	        title = file.split('/')[-1].lower()
	        if title in words_set:
	            title_location[title]=file

	homepage_sent = extract_wiki_sentences(title_location)
	# find the top 20 sentences from the above file
	word_homepage_sent = defaultdict(list)
	for w, sents in homepage_sent.items():
	    selected_sents = sents[:20]
	    word_homepage_sent[w]=selected_sents

    output_name = os.path.join(args.build_folder,'wiki_home_sent.json')
	json.dump(word_homepage_sent,open(output_name,'w'))

