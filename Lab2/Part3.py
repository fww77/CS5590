#%%
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tag import pos_tag
from nltk.util import ngrams
import re, collections

#1. Read the file and tokenize
input_tokenizer = nltk.data.load('SimpleDoc.txt')
sentences = sent_tokenize(input_tokenizer)
print (sentences)
print('\n')
words = word_tokenize(input_tokenizer)
print (words)

#2. Apply lemmatization on the words
print("\npart b \n")
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
lemmatization=[]
for word in words:
    lemmatization.append(lemmatizer.lemmatize(word))
print (lemmatization)


#3. Apply the bigram on the text
print("\npart c \n")
bigrams = list(ngrams(words,2))
print (bigrams)


#4. Calculate the word frequency (bi-gram frequency) of the words (bi-grams)
print("\npart d \n")
while '""' in bigrams:
    bigrams.remove('""')
while '[ ' in bigrams:
    bigrams.remove('[ ')
    while '] ' in bigrams:
        bigrams.remove('] ')
while '.' in bigrams:
    bigrams.remove('.')
while ',' in bigrams:
    bigrams.remove(',')
bigrams_count = collections.Counter(bigrams)
print (bigrams_count)

#5 Choose top five bi-grams that have been repeated most
print("\npart f \n")
FreqWord = bigrams_count.most_common(5)
print (FreqWord)
FreqWord = bigrams_count.most_common(1)
print (FreqWord)


#
#6 Find all the sentences with those most repeated bi-grams
print("\npart h \n")
String= ''.join(sentences)
Newstring=String.split(".")
word1=(FreqWord[0][0][0])
word2=(FreqWord[0][0][1])
word3= word1 + " " + word2
print(word3)
FreqSen = []
for i in range(len(Newstring)):
    if word3 in Newstring[i]:
        FreqSen.append(Newstring[i])
print (FreqSen)


#7 Extract those sentences and concatenate
print("\npart i \n")
s = []
s.append(FreqSen)
s = ''.join(FreqSen)
print(s)