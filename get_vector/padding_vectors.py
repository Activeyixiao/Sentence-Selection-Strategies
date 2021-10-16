import os
import json
from collections import defaultdict
import pickle
import torch
import numpy as np
from argparse import ArgumentParser

def folder2vec(folder):
	D = {}
	all_file = [os.path.join(folder,f) for f in os.listdir(folder)]
	for file in all_file:
		word_name = file.split('/')[-1]
		vec = torch.load(f)
		D[word_name]=vec
	return D


def pending_vec(source,target):
	new_d = {}
	for word,vecs in source.items():
		if word not in target:
			new_d[word]=vecs
		else:
			max_num = 20
			num = len(target[word])
			vec_ls = [vec.tolist() for vec in target[word]]
			if num < max_num:
				#print(word)
				pad_vecs = vecs[num:max_num]
				for v in pad_vecs:
					vec_ls.append(v)
			new_d[word]=torch.tensor(vec_ls[:20])
	return new_d

def pending_vec_single(source,target): 
	# this one is used for dictonary file which each word only have a single vector such as definition vector
	new_d = {}
	for word,vecs in source.items():
		if word not in target:
			new_d[word]=torch.tensor(vecs[0].tolist())
		else:
			new_d[word]=target[word]
	return new_d



if __name__ == '__main__':

	parser = ArgumentParser() 
	parser.add_argument('-s','--padding-source', help='source vectors files used to pad the missing word vectors', required=True)
	parser.add_argument('-t','--padding-target', help='target vectors file that might lose some word vectors', required=True)
	parser.add_argument('-b','--build-folder', help='location of output files',required=True)

	args = parser.parse_args()
	if not os.path.exists(args.build_folder):
		os.makedirs(args.build_folder)

	print('loading source dictionary...')
	if os.path.isfile(args.padding_source):
		source_d = pickle.load(open(args.padding_source,'rb'))
	elif os.path.isdir(args.padding_source):
		source_d = folder2dic(args.padding_source)

	print('loading target dictionary...')
	if os.path.isfile(args.padding_target):
		target_d = pickle.load(open(args.padding_target,'rb'))
	elif os.path.isdir(args.padding_target):
		target_d = folder2dic(args.padding_target)

	print('padding vectors...')
	if not 'def' in args.padding_target:
		result_d = pending_vec(source_d,target_d)
	else:
		result_d = pending_vec_single(source_d,target_d)

	output_name = os.path.join(args.build_folder,args.padding_target)
	pickle.dump(result_d,open(output_name,'wb'))
	print('Done')
