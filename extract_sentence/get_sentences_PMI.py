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


	"""This cell sample sentences from the word_pair_sentences"""
	word_neighbor_sentences = defaultdict(lambda:defaultdict(list))

	all_file= [os.path.join(args.corpus_dir,file) for file in os.listdir(args.corpus_dir)]
	print('there are',len(all_file),'words in the folder')
	for file in all_file:
		word_name = file.split('/')[-1]
		if word_name in words_set:
			with open(file) as inf:
				for line in inf.readlines():
					try:
						cols = line.strip().split('___')
						neighbor, sent = cols
						word_neighbor_sentences[word_name][neighbor].append(sent)
					except:
						print('missing sent:',sent)

	print('word_neighbor_sentences_dictionarty is ready with',len(word_neighbor_sentences.keys()),'words inside')
	
	word_neighbor_sent = defaultdict(list)
	w_count = 0
	for w in word_neighbor_sentences.keys():
		all_sents = [s for s_ls in word_neighbor_sentences[w].values() for s in s_ls]
		sampled_sent = set()
		while len(word_neighbor_sent[w])<20 and len(all_sents)>20:
			for k,vs in word_neighbor_sentences[w].items():
				sent = random.sample(vs,1)[0]
				sampled_sent.add(sent)
				word_neighbor_sent[w].append(sent)
				if len(word_neighbor_sent[w])>=20:
					break
		w_count += 1
		if w_count % 100==0:
			print(w_count,'words has been processed')


	print('word_neighbor_sent_dictionarty is ready with',len(word_neighbor_sent.keys()),'words inside')

	file_name = os.path.join(args.build_folder,'wiki_PMI_sent.json')
	json.dump(word_neighbor_sent,open(file_name,'w'))
	print('Done')