Current word embedding learning technology mainly relay on randomly sampling sentences from corpus for each word, it is suspect that if these random sentences can really capture semantic properties of word given that many sentence that mentioning a word doesn't give any clue of its commonsense property.

This project aims at selecting better sentences for learning word embedding that better at capturing semantic properties.

Three kinds of strategies will be proposed and tested in this project:
  1.Using wikipedia structure
  2.Using PPMI value to extract revevant word pair and therefore find more informative sentences
  3.Using topic-model (Latent Dirichlet Allocation)
  
The baseline will be the vectors learned from random selected sentences.

Evaluation of these strategies will be taken on :
  1.word-sensen-classification (BD,SS,MC,CSLB)
  2.relation classificaton (ConceptNet, BLESS, Hperlex)

