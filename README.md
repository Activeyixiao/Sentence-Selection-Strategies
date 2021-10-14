
# Sampling-sentences-for-better-word-embeddings

## Introduction: better word-vectors require better sentences

Current word embedding learning technology commonly find random sentences to learn a representation for a word, it is not guaranteed that random selected sentences can capture semantic properties of a word given that a sentence in which a word in may not give any clue of its commonsense properties. (e.g. the banana is yellow and sweet). Even if increasing coverage of the sentence selection can cover most useful sentences, applying average upon them inevitably loose some relevant information, not mentioning its inefficiency. If we could find only a few representative sentences for each words and learn a representation, this representation might be more sensitive and effective to some downstream NLP tasks that requires capturing semantic properties.

## Method: how to get better sentences:

Motivied by above idea, we proposed that the following strategies could help us find less and better sentences than random sampling.

  - A.Using wikitionary dataset and finding definition sentence for each word
  - B.Using wikipedia structure (sentences from first section, or first papragraph from the wiki-homepage of each word)
  - C.Using PPMI value to extract most revevant word pair and seletct sentences that mentioning both words.
  - D.Using GenericsKB datatset and selecting relevant sentences for each word 
  
After obtaining those sentence for each word, we could use pre-trained language model (BERT or RoBERTa) to extract its hidden layers representation for that word in each sentence. Then we sum these vectors up and divided by the number of it, taking plain-average of these vectors as the final representation of that word. The baselines that we want to comare with are the vectors learned from a randomly selected sentence.

## Doing it step by step:

### Requirements
- Python3
- Numpy
- Torch
- pickle
- NLTK

### Step 1: Preparing a vocabulary list

- The code requires a text file containing the vocabulary, i.e. the set of words for which vector representations need to be obtained. This vocabulary is encoded as a plain text file, with one word per line.

- The vocabulary corresponding to the experiments from our paper can be downloaded here: https://cf-my.sharepoint.com/:t:/g/personal/wangy306_cardiff_ac_uk/EXg5FWbRhLVDlXrPAd0vwCUBNkMTiJGiSRTFQtaYtOycaA?e=ev3epd

- In this case, the vocabulary consists of all words from the four evaluation dataset: [the extended McRae feature norms](https://github.com/mbforbes/physical-commonsense), [CSLB](https://cslb.psychol.cam.ac.uk/propnorms#:~:text=The%20Centre%20for%20Speech%2C%20Language,feature%20representations%20of%20conceptual%20knowledge.), [WordNet_Supersenses](https://wordnet.princeton.edu/), and [BabelNet domains](http://lcl.uniroma1.it/babeldomains/#:~:text=BabelDomains%20is%20a%20unified%20resource,the%20Wikipedia%20featured%20articles%20page.)

### Step 2: Selecting sentences for each word

