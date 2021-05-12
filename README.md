
# Sampling-sentences-for-better-word-embeddings

## Motivation:

Current word embedding learning technology commonly find random sentences to learn a vector for each word, it is suspect that if these random sentences can really capture semantic properties of a word given that many sentences that mentioning that word may not give any clue of its commonsense properties. (e.g. the banana is yellow and sweet). Even if the coverage of the random selection is large enough to include all useful sentences, using average upon them inevitably loose some sensitive information, not mentioning the inefficiency of this strategy.

If we could find only a few representative sentences for each words and learn a embedding without applying average upon them, the resulted vectors might be more sensitive to some downstream NLP tasks that focus on semantic properties beside that these vectors can be easy to learn.

## Method:

Motivied by above idea, we proposed that the following strategies could help us find less and better sentences than random sampling.

  A0.Using wiki-defintion sentence for each word (random sentence will be used if some word don't have definition sentence)
  A1.Using wikipedia structure (use only the fixed number of sentences from first section, or first papragraph of home page of each word)
  A2.Using PPMI value to extract most revevant word pair and use sentences that mentioning both words as samples.
  A3.coming soon...
  
After obtaining those sentence for each word, we could use pre-trained language model (BERT or RoBERTa) to extract its final layer representation for that word per sentence. The result will be that each word gain several vectors corresponding to its sample sentences. Rathan than taking plain-average or weighted-average of these vectors, we propose to directly take them as input to downstram NLP tasks and use CNN model (kernal size = 1) to do feature extraction. The advantage of using CNN model is that it can find the most suitable vector after max-pooling and thus make best use of that vector for evaluaton tasks.  

The baselines to compete will be the following options:
  B0.vectors learned from a random selected sentence (to compare with the A0).
  B1.vectors learned from several random selected sentences after averaging them (to compare with A1 or A2, the number of sample sentences is also same)
  B2.vectors learned from 500 random sentences after average them (to confirm if less and better sample sentences after CNN filter are better than more and messy sample sentence after averaging)
  
## Experiment and evaluation:
  1.word-sensen-classification (BD,SS,MC,CSLB)
  2.relation classificaton (ConceptNet, BLESS, Hperlex)

