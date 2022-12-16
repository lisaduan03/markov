"""
storing words that are used next = distribution in the text conditional on 
preceding word
"""

#import text file and split into single words with punctuation 
import numpy as np
trump = open('speech.txt', encoding='utf8').read()

corpus = trump.split()

# use lazy evalution and yield generator object instead of acutally filling our
# memory with every pair of words
def make_triples(corpus):
    for i in range(len(corpus)-2):
        yield (corpus[i], corpus[i+1], corpus[i+2])
        
triples = make_triples(corpus)

# empty dictionary and fill with words. If first word is already a key, append 
# next word. Otherwise, initialize a new entry
word_dict = {}
for word_1, word_2, word_3 in triples:
    if word_1 in word_dict.keys():
        word_dict[word_1].append(word_2)
    else:
        word_dict[word_1] = [word_2, word_3]

# pick random word to kick off chain and choose number of words to simulate
first_word = np.random.choice(corpus)
chain = [first_word]   
n_words = 30

# after first word, every word in chain is sampled randomly from list of words
# which have followed that word in Trump's actual speeches
for i in range(n_words):
    chain.append(np.random.choice(word_dict[chain[-1]]))

# join command returns chain as string 
' '.join(chain)