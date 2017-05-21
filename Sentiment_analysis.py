#Python 3
import re #Please check that you have these!
from os import listdir
import numpy as np #Only end of recommendation system is affected

# Full sentiment analysis of a single review
file1 = 'pos/6_10_tt0100680.txt' #Check other 
f = open(file1,'r')

vocab = {}
for line in f:
    line = line.strip().lower()
    words = re.split(' |, |: |!|\.|"|\(|\)|\?|/|;|>|<',line)
    for word in words:
        if word == '':
            continue
        vocab[word]=vocab.get(word,0)+1
f.close()

# Load polarity words, adapted from http://sentiwordnet.isti.cnr.it/
f = open('polarity_words_uniq.csv','r')
i = 0
pol_words = {} #Dictionary of polarity words: {pol_word1:polarity1, pol_word2:polarity2,...}
next(f) #Skip header
for line in f:
    line = line.strip()
    line = line.split(',')
    pol_words[line[0]] = np.sign(float(line[1])) #+1 is for positive words, -1 is for negative words

vote = 0
for word in vocab.keys():
    polarity = pol_words.get(word,0)
    vote += polarity
if vote>0:
    print("Positive review, vote =",vote)
elif vote == 0:
    print("Neutral review")
else:
    print("Negative review, vote =",vote)

##### CLASSIFY ALL REIEWS ####
# Using regular expressions for tokenization
fold = 'pos/'
n_pos = 0 #Number of positive reviews
n_files = 0 #Total number of files
for file in listdir(fold): #Get the name of each file in fold
    n_files += 1
    if n_files>5: #Comment this
        break     #Comment this
    file1 = fold + file
    vote = 0 #The total vote for each review
    try:
        f = open(file1,'r')
        for line in f:
            print(line) #Comment this
            line = line.strip().lower()
            words = re.split(' |, |: |!|\.|"|\'|\(|\)|\?|/|;|>|<',line)
            for word in words:
                polarity = pol_words.get(word,0)
                vote += polarity
        print('\n') #Comment this
        if vote>0:
            print("Positive review, vote =",vote) #Comment this
            n_pos += 1
        elif vote == 0:
            print("Neutral review") #Comment this
            pass
        else:
            print("Negative review, vote =",vote) #Comment this
            pass
        print('\n') #Comment this
    except:
        continue

#Check the 4th review with the vote -14. It is indeed difficult to say from the text that this is a positive review.
print("Classifier accuracy is",n_pos/n_files)