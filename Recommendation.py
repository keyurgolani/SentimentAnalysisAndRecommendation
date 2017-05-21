from os import listdir
import re

# Form a dictionary for each movie with all reviews
fold = 'pos/'
all_reviews = {}  # Dictionary: {movID1:[rev1,rev2,...], movID2:[rev1,rev2,...]}
for file in listdir(fold):
    movID = re.split('_|\.', file)[2]  # file format: [revID]_[score]_[movID].txt
    l = all_reviews.get(movID, [])  # list of all previously seen reviews for this movID
    try:
        f = open(fold + file, 'r')
        review = ''
        for line in f:
            review = review + '.' + line  # Join all lines of a review, separate them by "."
        l.append(review)  # Append each review to a list of reviews
        all_reviews[movID] = l  # Update list of reviews for this movID
        f.close()
    except:
        continue


tot = 0
top_rev_ids = []  # IDs of movies with more than 10 reviews
top_rev_bow = []  # bow - "bag of words" for these movies
for movID, reviews in all_reviews.items():
    if len(reviews) > 10:  # Use only movies with more than 10 reviews for recommendation
        top_rev_ids.append(movID)
        bow = {}
        for review in reviews:
            review = review.strip().lower()
            words = re.split(' |, |: |!|\.|"|\'|\(|\)|\?|/|;|>|<', review)
            for word in words:
                bow[word] = bow.get(word, 0) + 1
                # Word's count in all previously seen reviews for this movID
        top_rev_bow.append(bow)

full_vocab = {}  # Vocabulary of all words in positive reviews: {word1:count1, word2:count2, ...}
for bow in top_rev_bow:
    for word, count in bow.items():
        full_vocab[word] = full_vocab.get(word, 0) + count

high_freq_words = set(sorted(full_vocab, key=full_vocab.get, reverse=True)
                      [:88])  # High frequency words

# Polarity words
f = open('polarity_words_uniq.csv', 'r')  # Adapted from http://sentiwordnet.isti.cnr.it/
pol_words = {}
next(f)  # To skip header row
for line in f:
    line = line.strip()
    line = line.split(',')
    polarity = float(line[1])
    pol_words[line[0]] = polarity
f.close()

high_pol_words = set([k for k, v in pol_words.items() if v >= 0.5 or v <= -0.5])

vocab = set(full_vocab.keys()) - high_pol_words - high_freq_words
# Vocabulary for use in recomendation system

vocab_pos = dict(zip(vocab, range(len(vocab))))
# Dictionary of vocabulary words and their positions in vocab_vec: {word1:pos1, word2:pos2,...}

vocab_vec = sorted(vocab_pos, key=vocab_pos.get, reverse=False)  # List of vocabulary words in order

import numpy as np
n_top_rev_mov = len(top_rev_ids)  # Number of top reviewed positive movies
n_vocab = len(vocab_pos)  # Number of all words in vocabulary
# Initially mov_vocab_matr is set to 0s
mov_vocab_matr = np.zeros((n_top_rev_mov, n_vocab), dtype=np.int)
# Rows correspond to each movie in top_rev_bow; columns correspond to each word in vocab_vec

for movInd, bow in enumerate(top_rev_bow):
    for word in bow.keys():
        try:
            # If word has been seen in a review, set to 1
            mov_vocab_matr[movInd, vocab_pos[word]] = 1
        except:
            continue

f = open('neg/4_4_tt0047200.txt', 'r')  # Open negative review of an unsatisfied customer
vocab_vec_neg_mov = np.zeros(n_vocab, dtype=np.int)  # vector of 0s for each word in vocabulary
for line in f:
    line = line.strip().lower()
    words = re.split(' |, |: |!|\.|"|\'|\(|\)|\?|/|;|>|<', line)
    for word in words:
        try:
            vocab_vec_neg_mov[vocab_pos[word]] = 1  # if word has been seen in this review, set to 1
        except:
            continue
f.close()

print([top_rev_ids[i] for i in np.argsort(-np.dot(mov_vocab_matr, vocab_vec_neg_mov))][:5])
# Recommended movies (sorted from the most similar to least similar)
