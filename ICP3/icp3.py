# import collections
# MyDictionary=input("please enter a string:")
# Freq = {}
# def split_string(MyDictionary):
#     words= MyDictionary.split()
#     for word in words:
#         print(word)
#         Freq[word] += 1
#     print(Freq)
#
# if __name__ == "__main__":
#     split_string(MyDictionary)
#Question 1a
# MyDictionary=input("please enter a string: ")
# def split_string(MyDictionary):
#   my_string = MyDictionary.split()
#   my_string.sort()
#   MyDict = {}
#   for item in my_string:
#     MyDict[item] = my_string.count(item)
#   print(MyDict)
# if __name__ == "__main__":
#     split_string(MyDictionary)


#
# s = "Python Program "
# f={s}
# type(f)
# print(f)
# string =input("please enter string: ")
# list(string)
# set={string}
# type(set)
# print(set)
# vowels=0
# for i in string:
#     if(i=="a" or i =="A" or i=="e" or i=="E" or i=="i" or i=="I"
#             or i=="o" or i=="O" or i=="u" or i=="U"):
#         vowels +=1
# print("number of vowels = ")
# print(vowels)

# # #2
import urllib
import urllib.request
from bs4 import BeautifulSoup
import os
html_text = urllib.request.urlopen("https://en.wikipedia.org/wiki/List_of_state_and_union_territory_capitals_in_India")
html_doc= html_text
soup = BeautifulSoup(html_doc, "html.parser")
print(soup.title.getText())

for link in soup.find_all('a'):
    href = link.get('href')
    if href is not None and "http" not in href:
        href = "https://en.wikipedia.org" + href

    print(href)


table = soup.find('table','wikitable sortable plainrowheaders')


for row in table.find_all('tr'):
    print()
    for cell in row.find_all('td'):
        print (cell.getText())

