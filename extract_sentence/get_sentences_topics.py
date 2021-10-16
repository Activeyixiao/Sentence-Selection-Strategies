import os
from collections import defaultdict
from argparse import ArgumentParser
import random
import json


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


	"""This cell sample sentences from the topic_specific_sentences"""
	word_topic_sent_20 = defaultdict(list)

	all_folder = [os.path.join(args.corpus_dir,file) for file in os.listdir(args.corpus_dir)]
	print('there are',len(all_folder),'words in the folder')
	for folder in all_folder:
	    word_name = folder.split('/')[-1]
	    num_s=0
	    if word_name in words_set:
	        try:
	            all_topic = [os.path.join(folder,f) for f in os.listdir(folder)]
	            used_set = set()
	            while num_s<20:
	                for t in all_topic:
	                    if num_s<20:
	                        with open(t) as inf:
	                            title = inf.readline()
	                            all_line = inf.readlines()
	                            sent = random.sample(all_line,1)[0]
	                            while sent in used_set and len(all_line)>=10:
	                                sent = random.sample(all_line,1)[0]
	                            used_set.add(sent)
	                            sent = sent.strip().split('___')[1]
	                            word_topic_sent_20[word_name].append(sent)
	                    else:
	                        break

	        except:
	            print(word_name)
	            pass

	file_name = os.path.join(args.build_folder,'wiki_topic_sent.json')
	json.dump(word_topic_sent_20,open(file_name,'w'))