
# #  Question 1: method 1
# myList1=["PHP", "Exercises", "Backend"]
# longestWord= max(myList1,key=len)
# print(len(longestWord),longestWord)
#
# # Question 1: method 2
# def GetLongestWord(myList1):
#     myList2 =[]
#     for n in myList1:
#         myList2.append((len(n),n))
#     myList2.sort()
#     return myList2[-1]
# print(GetLongestWord(["PHP", "Exercises", "Backend"]))




# # Question 2
# from collections import Counter
# WordList=['dog','dog','cat','cat','cat','dog','car','tree']
# Frequency= Counter(WordList)
# print(Frequency)

# #Question 3: method 1
# import string
# myString=("DeepLearning python machineLearning")
# words=myString.split()
# numberOfWords=len(words)
# print(numberOfWords)
#
# #Question 3: method 2
# word_count =0
# filename = '‎filefortest.txt'
# with open(filename, "r") as file:
#     for line in file:
#         word_count +=len(line.split())
#     print("number of words: ", word_count)
#

# Question4

heightinp = int(input("Enter the height of the board: "))
widthinp = int(input("Enter the width of the board: "))


def board_draw(heightinp, widthinp):
    for i in range(heightinp):
        print(" ---"* widthinp)
        print("|   "*(widthinp+1))
    print (" ---"*(widthinp))


if __name__ == "__main__":
    board_draw(heightinp, widthinp)

