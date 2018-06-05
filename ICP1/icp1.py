#Question1
import sys
print (sys.version)

#  Question 2a
Fname=input("please enter your first name:")
Lname=input("please entr your last name:")
print(Lname + Fname)

# Question 2b
FirstNumber = input("please enter your first number:")
SecondNumber = input("please enter your second number:")
number1 = int(FirstNumber)
number2 = int(SecondNumber)
Output = divmod(number1,number2)
print (Output)

# Question 3
import random
guessNumber =0
number =random.randint(1,15)
while guessNumber <15:
    guess =input()
    guess =int(guess)
    guessNumber =guessNumber +1
    if guess <number:
        print ("your number is lower than required.")
    if guess >number:
        print("your number is higher than required")
    if guess == number:
        break
if guess == number:
        print("Your answer is PERFECT!! Congratulations!!")

