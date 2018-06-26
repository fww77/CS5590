import urllib.request
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize,wordpunct_tokenize,sent_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.util import ngrams
from nltk import ne_chunk
import fileinput
# urllib.request.urlretrieve("https://en.wikipedia.org/wiki/Python_(programming_language)","input.txt")
html_text = urllib.request.urlopen("https://en.wikipedia.org/wiki/Python_(programming_language)")
soup = BeautifulSoup(html_text, "html.parser")
text=soup.get_text()
# print (text)
with open('input.txt','w',encoding='utf-8') as output:
    output.write(text)

#Tokenization
print("this is for Tokenization")
input_tokenizer = nltk.data.load('input.txt')
sentences = sent_tokenize(input_tokenizer)
print (sentences)
print('\n')
words = word_tokenize(input_tokenizer)
print (words)


#
# # # Stemming
print("this is for Stemming")
stemmer = SnowballStemmer("english")
output_stemming = stemmer.stem(text)
print (output_stemming)
#
# # # #POS
print("this is for POS")
nltk.download('averaged_perceptron_tagger')
output_POS = []
# for tokens in words:
#     output_POStags.append(nltk.pos_tag(tokens))
# print (output_POStags)
output_POS.append(nltk.pos_tag(words))
print (output_POS)


# # #
# # #Lemmatization
print("this is for Lemmatization")
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
lemmatization=[]
for word in words:
    lemmatization.append(lemmatizer.lemmatize(word))
# output_lemmatization = lemmatizer.lemmatize(words)
print (lemmatization)

# #
# #Trigram
print("this is for Trigram")
trigrams = list(ngrams(words,3))
print (trigrams)
# # #
# # # #Named Entity Recognition
print("this is for Named Entity Recognition(NER)")
nltk.download('maxent_ne_chunker')
nltk.download('words')
NER = ne_chunk(pos_tag(wordpunct_tokenize(input_tokenizer)))
print (NER)