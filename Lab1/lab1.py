# Question1:
# password=input("please enter your password:")
# def Validate(password):
#         if len(password)<6:
#             print("error! try a longer password")
#         if len(password)>16:
#             print("error! try a shorter password")
#         if not any(i.isdigit() for i in password):
#             print("error! password must contain one number")
#         if not any(i.islower()for i in password):
#             print("error! password must contain one lower character")
#         if not any(i.isupper()for i in password):
#             print("error! password must contain one upper character")
#         if "$" not in password and "@" not in password and "!" not in password and "*" not in password:
#             print("error! password must contain one special character")

# if __name__== "__main__":
#     Validate(password)
# Question1b

import re  # used to search passwords

# Prompt User for username
username = str(input("Choose your username: "))
# Prompt User for password
password = str(input("Choose your password: "))
password_pass = 0
# While loop will only exit when all requirements have been met
while password_pass != 6:
    # Test Min Length
    if len(password) >= 6:
        password_pass += 1
    else:
        print("Error 1: The password you have chosen is too short. Try Again.")
        password_pass = 0
        password = str(input("Choose new password: "))
        continue

    # Test Max Length
    if len(password) <= 16:
        password_pass += 1
    else:
        print("Error 2: The password you have chosen is too long. Try Again.")
        password_pass = 0
        password = str(input("Choose a new password: "))
        continue

    # Test Special Character Present
    if re.search('[!@#$%&]', password):
        password_pass += 1
    else:
        print("Error 3: The password must have a spec char. Try Again.")
        password_pass = 0
        password = str(input("Choose a new password: "))
        continue
    # Test Capital Letter Present
    if re.search('[A-Z]', password):
        password_pass += 1
    else:
        print("Error 4: The password must have at least one capital letter")
        password_pass = 0
        password = input("Choose a new password: ")
        continue
    # Test Lowercase Letter Present
    if re.search('[a-z]', password):
        password_pass += 1
    else:
        print("Error 5: The password must have at least one lowercase letter")
        password_pass = 0
        password = input("Choose a new password: ")
        continue
    if re.search('[0-9]',password):
        password_pass += 1
    else:
        print("Error 6: The password must have at least one number")
        password_pass = 0
        password = input("Choose a new password: ")
        continue

print("Thank You. Your username and password has been saved.")
exit()

#2 Question2
# mySentence=input("please enter your sentence:")
# myList=mySentence.split()
# def WordGame(myList):
#     length=len(myList)
#     if len(myList)%2==0:
#         print(myList[int(length/2-1)],myList[int(length/2)])
#     else:
#         print(myList[len])
#     longestWord= max(myList,key=len)
#     print(longestWord)
#     print(' '.join(w[::-1] for w in myList))
    # if need convert to lower case,just convert the string before print usimg lower() function


# if __name__== "__main__":
#     WordGame(myList)
#
#3 Question3
# myList=[-25,-10,-7,-3,2,4,8,10]
# if len(myList)<3:
#     print("please input a list with more than 3 numbers:")
# for i in range(len(myList) - 2):
#     for j in range(i+1,len(myList)-1) :
#         for k in range(j+1, len(myList)):
#             if myList[i] +myList[j]+myList[k] ==0:
#                 print([myList[i],myList[j],myList[k]])


#4 Question4
 # two sets: Set A contains all students select "python" and set B contains all students select "Web Application"
# import random as rand
# from people import Person
#
# first_names = ["Thomas", "John", "Mike", "Liz", "Joan", "Hattie", "Emma"]
# last_names = ["Smith", "Jones", "Brown", "Miller", "Tran", "Cook", "Scott"]
# python = []
# web = []
#
#
# def random_string(input: list) -> str:
#     num = rand.randint(0, len(input) - 1)
#     return input[num]
#
#
# def generate_random_students(fname: list, lname: list, number_of_students: int) -> list:
#     """creates a list of persons with random names"""
#     lst = []
#     for _ in range(0, number_of_students):
#         lst.append(Person(random_string(fname), random_string(lname)))
#     return lst
#
#
# def random_partion(class1: list, class2: list, students: list):
#     """randomly places a student in non one or both class list"""
#     for student in students:
#         num = rand.randint(0, 4)
#         if num == 0:
#             pass
#         elif num == 1:
#             class1.append(student)
#
#         elif num == 2:
#             class2.append(student)
#
#         elif num == 3:
#             class1.append(student)
#             class2.append(student)
#     return
#
#
# def print_student(students):
#     """ prints the students name and its memory location"""
#     for student in students:
#         print(student.get_full_name() + " " + str(student))  #
#
#
# student_list = generate_random_students(first_names, last_names, 10)
#
# random_partion(python, web, student_list)
# # turns all list into a set
# python_set = set(python)
# web_set = set(web)
# student_set = set(student_list)
# # finds the students in both classes
# common_set = python_set.intersection(web_set)
# # finds the student  who have no classes
# no_class = student_set.difference(student_set.difference(python_set), web_set)
#
# print()
# print("common classes")
# print()
# print_student(common_set)
# print()
#
# print("No class")
# print()
# print_student(no_class)
# print()
#
#
# class Flight:
#
#     def __init__(self, flight_number, from_location, to_location, depart_time, arrive_time):
#
#         # Type flight_number: Integer
#         # Type from_location: String
#         # Type to_location: String
#         # Type depart_time: String ('mm/dd/yyyy hh:mm AM/PM')
#         # Type arrive_time: String ('mm/dd/yyyy hh:mm AM/PM')
#
#         self.__num = flight_number
#         self.__from = from_location
#         self.__to = to_location
#         self.__depart = depart_time
#         self.__arrive = arrive_time
#
#     def __str__(self):
#
#         # Return Type: String
#
#         out = 'Flight Number: ' + str(self.__num) + ' (' + self.__from + ' - ' + self.__to + ')\n'
#         out = out + 'Depart: ' + self.__depart + ', Arrive: ' + self.__arrive
#         return out
#
#
# class Person:
#
#     def __init__(self, last_name, first_name, date_of_birth):
#
#         # Type last_name: String
#         # Type first_name: String
#         # Type date_of_birth: String ('mm/dd/yyyy')
#
#         self.__last = last_name
#         self.__first = first_name
#         self.__dob = date_of_birth
#
#     def __str__(self):
#
#         # Return Type: String
#
#         return self.__first + ' ' + self.__last
#
#
# class Employee(Person):
#
#     def __init__(self, last_name, first_name, date_of_birth, employee_number):
#
#         # Type last_name: String
#         # Type first_name: String
#         # Type date_of_birth: String ('mm/dd/yyyy')
#         # employee_number: Integer
#
#         super().__init__(last_name, first_name, date_of_birth)
#         self.__id = employee_number
#
#     def book_air_ticket(self, traveler, flight):
#
#         # Type traveler: Passenger
#         # Type flight: Flight
#         # Return Type: AirTicket
#
#         return traveler.book_air_ticket(self, flight)
#
#
# class Passenger(Person):
#
#     def __init__(self, last_name, first_name, date_of_birth):
#
#         # Type last_name: String
#         # Type first_name: String
#         # Type date_of_birth: String ('mm/dd/yyyy')
#
#         super().__init__(last_name, first_name, date_of_birth)
#         self.__itineraries = []
#
#     def book_air_ticket(self, agent, flight):
#
#         # Type agent: Employee
#         # Type flight: Flight
#         # Return Type: AirTicket
#
#         ticket = AirTicket(self, agent, flight)
#         self.__itineraries.append(ticket)
#         return ticket
#
#     def get_tickets(self):
#
#         # Return Type: String
#
#         out = ''
#         for ticket in self.__itineraries:
#             out = out + str(ticket.verify_payment()) + '\n'
#         return out[:-1]
#
#
# class Itinerary:
#
#     def __init__(self, traveler):
#
#         # Type traveler: Passenger
#
#         self.__owner = traveler
#
#     def get_traveler(self):
#
#         # Return Type: Passenger
#
#         return self.__owner
#
#
# class Payment:
#
#     def __init__(self, traveler, agent):
#
#         # Type traveler: Passenger
#         # Type agent: Employee
#
#         self.__payer = traveler
#         self.__payee = agent
#         self.__status = 'Pending'
#
#     def verify(self):
#
#         # Return Type: Void
#
#         self.__status = 'Verified'
#
#     def get_status(self):
#
#         # Return Type: Void
#
#         return self.__status
#
#
# class AirTicket(Itinerary, Payment):  # Multiple Inheritance
#
#     def __init__(self, traveler, agent, flight):
#
#         # Type traveler: Passenger
#         # Type agent: Employee
#         # Type flight: Flight
#
#         Itinerary.__init__(self, traveler)
#         Payment.__init__(self, traveler, agent)
#         self.__air = flight
#
#     def verify_payment(self):
#
#         # Return Type: AirTicket
#
#         self.verify()
#         return self
#
#     def __str__(self):
#
#         # Return Type: String
#
#         out = 'Passenger Name: ' + str(self.get_traveler()) + '\n'
#         out = out + str(self.__air) + '\n'
#         out = out + self.get_status()
#         return out
#
#
# def main():
#
#     f1 = Flight(582, 'Kansas City, MO', 'Omaha, NE', '06/15/2018 08:00 am', '06/12/2018 09:00 am')
#     f2 = Flight(2354, 'Chicago, IL', 'Tampa, FL', '06/26/2018 01:30 pm', '06/20/2018 04:40 pm')
#     f3 = Flight(6666, 'San Francisco, CA', 'Houston, TX', '07/04/2018 09:00 pm', '07/04/2018 11:50 pm')
#
#     p1 = Passenger('Fei', 'Wu', '06/21/1990')
#     p2 = Passenger('Yunlong', 'Liu', '06/19/1993')
#
#     a1 = Employee('Ting', 'xia', '10/15/1990', 23606)
#     a2 = Employee('Emily', 'Chen', '05/13/1989', 23832)
#
#     a2.book_air_ticket(p1, f1)
#     a1.book_air_ticket(p2, f2)
#     p1.book_air_ticket(a2, f3)
#
#     print(p1.get_tickets())
#     print(p2.get_tickets())
#
#
# if __name__ == '__main__':
#     main()

# # 6 Question6
# import numpy
#
# def random_array_generator():
#     return numpy.random.randint(0, 21, 15).tolist()
#
#
# def most_frequent_num(array):
#     count, out = [0] * 21, ''
#     for num in array:
#         count[num] = count[num] + 1
#     most_freq = sorted([i for i in range(0, 21) if count[i] == max(count)])
#     if len(most_freq) == 1:
#         out = out + 'Most frequent item in the list is ' + str(most_freq[0]) + '.'
#     else:
#         out = out + 'Most frequent items in the list are '
#         for index, num in enumerate(most_freq):
#             if index == len(most_freq) - 1:
#                 out = out + 'and ' + str(num) + '.'
#             else:
#                 out = out + str(num) + ', '
#     return out


# def main():
#
#     array = random_array_generator()
#     print(array)
#     print(most_frequent_num(array))
#
#
# if __name__ == '__main__':
#     main()
