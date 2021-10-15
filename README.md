
# Build better word-representations by selecting better sentence

## Introduction

:smiley: We believe high-quality sentences lead to high-quality word representation. Current word embedding learning technology commonly find random sentences to learn a representation for a word, it is not guaranteed that random selected sentences can capture semantic properties of a word given that a sentence in which a word in may not give any clue of its commonsense properties. (e.g. the banana is yellow and sweet). Even if increasing coverage of the sentence selection can cover most useful sentences, applying average upon them inevitably loose some relevant information, not mentioning its inefficiency. If we could find only a few representative sentences for each words and learn a representation, this representation might be more sensitive and effective to some downstream NLP tasks that requires capturing semantic properties.

## Method: how to get better sentences:

Motivied by the above idea, we proposed that the following strategies could help us finding better sentences than random selecting.

  - A.Using wikitionary dataset and finding definition sentence for each word
  - B.Using wikipedia structure (sentences from first section, or first papragraph from the wiki-homepage of each word)
  - C.Using PPMI value to extract most revevant word pair and seletct sentences that mentioning both words.
  - D.Using topic model to select sentences
  - E.Using GenericsKB datatset and selecting relevant sentences for each word 
  
After obtaining those sentence for each word, we could use pre-trained language model (BERT or RoBERTa) to extract its hidden layers representation for that word in each sentence. Then we sum these vectors up and divided by the number of it, taking plain-average of these vectors as the final representation of that word. The baselines that we want to comare with are the vectors learned from a randomly selected sentence.

## Doing step by step:

#### Requirements
- Python3
- Numpy
- Torch
- pickle
- NLTK

### Preparing a vocabulary list

- The code requires a text file containing the vocabulary, i.e. the set of words for which vector representations need to be obtained. This vocabulary is encoded as a plain text file, with one word per line.

- The vocabulary corresponding to the experiments from our paper can be downloaded here: https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/EXg5FWbRhLVDlXrPAd0vwCUBNkMTiJGiSRTFQtaYtOycaA?e=ev3epd

- In this case, the vocabulary consists of all words from the four evaluation dataset: [the extended McRae feature norms](https://github.com/mbforbes/physical-commonsense), [CSLB](https://cslb.psychol.cam.ac.uk/propnorms#:~:text=The%20Centre%20for%20Speech%2C%20Language,feature%20representations%20of%20conceptual%20knowledge.), [WordNet_Supersenses](https://wordnet.princeton.edu/), and [BabelNet domains](http://lcl.uniroma1.it/babeldomains/#:~:text=BabelDomains%20is%20a%20unified%20resource,the%20Wikipedia%20featured%20articles%20page.)

### Selecting sentences

##### Required documents:
- vocabulary list: "all_words.txt"
- [wikipedia-file](https://doi.org/10.5281/zenodo.5570579)
- [wikipedia-introduction-file](https://doi.org/10.5281/zenodo.5570561)
- [wikipedia_homepages](https://doi.org/10.5281/zenodo.5570854)
- wikitionary-dataset: "data/all_wiktionary_onlynouns_most-freq-sense.csv"
- [genericsKG-dataset](https://allenai.org/data/genericskb)
- [word_topic(25)_sentences](https://doi.org/10.5281/zenodo.5570983)

##### obtaining sentences
- selecting random sentences from wikipedia: 
  - `python3 ./extract_sentence/get_sentences(random).py --words-file WORD_LIST --corpus-file SOURCE_FILE --intermiate-folder FOLDER_NAME --build-folder FOLDER_NAME --truncate TRUE_OR_FALSE`
- selecting sentences from introduction part of wikipedia
  - `python3 ./extract_sentence/get_sentences(random).py --words-file WORD_LIST --corpus-file SOURCE_FILE --intermiate-folder FOLDER_NAME --build-folder FOLDER_NAME --truncate TRUE_OR_FALSE`
- selecting sentences from wikitionary
  - `python3 ./extract_sentence/get_sentences(def_generics).py --words-file WORD_LIST --corpus-file SOURCE_FILE --source NAME_OF_SOURCE --build-folder FOLDER_NAME`
- selecting sentences from GenericsKG
  - `python3 ./extract_sentence/get_sentences(def_generics).py --words-file WORD_LIST --corpus-file SOURCE_FILE --source NAME_OF_SOURCE --build-folder FOLDER_NAME`
- selecting sentences from wiki-homepage
  - `python3 ./extract_sentence/get_sentences(wiki_homepage).py --words-file WORD_LIST --corpus-file SOURCE_FILE --build-folder FOLDER_NAME`
- selecting sentence using topic model (LDA)
  - `python3 ./extract_sentence/get_sentences(topics).py --words-file WORD_LIST --corpus-file SOURCE_FILE --build-folder FOLDER_NAME` 
- selecting sentences using PMI score of words co-occurrence
  - `python3 PMI_sentence.py`

### Getting vectors

##### Required documents:
- json files (dictionary that map word to its 20 sentences) as output from the above step.
##### Word vectors as hidden layers of langugae models (BERT, roBERTa)
- getting masked vectors:
  `python3 ./get_vector/run_mask.py -i <json file containing words-sentences-mapping> -s <max_seq_length> -b <batch_size> -o <out_dir: mask/unmask + LM_model_version+ json_file> -v <LM_version> -g <use GPU or not>`
- getting unmasked vectors:
  `python3 ./get_vector/run_unmask.py -i <json file containing words-sentences-mapping> -s <max_seq_length> -b <batch_size> -o <out_dir: mask/unmask + LM_model_version+ json_file> -v <LM_version> -g <use GPU or not>`
  
### Vector evaluation
- Evaluation task is semantic properties classification
- We evaluate our generated vectors on four dataset: [MC](https://github.com/mbforbes/physical-commonsense),[CSLB](https://cslb.psychol.cam.ac.uk/propnorms#:~:text=The%20Centre%20for%20Speech%2C%20Language,feature%20representations%20of%20conceptual%20knowledge.),[SS](https://wordnet.princeton.edu/),[BD](http://lcl.uniroma1.it/babeldomains/#:~:text=BabelDomains%20is%20a%20unified%20resource,the%20Wikipedia%20featured%20articles%20page)
- run evaluation experiment:
  `python3 ./evaluation/run_network.py -dataset <MC/SS/CSLB/BD> -embed_path <location of generated vector pickle file> -vector_name <bert_base,bert_large,roberta_base,roberta_large> -vector_type <definition/wiki_introduction/wiki_homepage/random/length_n/generics/topics/PMI> -in_features <768/1024> -batch_size <n>
`
