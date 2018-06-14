# Question 1
class employee:
    count=0
    def __init__(self,family, salary, department):
        self.family = family
        self.salary = salary
        self.department = department
        employee.count+=1
    def getInfor(self):
            return self.family, self.salary, self.department
    def avg_salary(self):
        return sum(self.salary)/employee.count

# Create a Fulltime Employee class and it should inherit the properties of Employee class
if __name__== "__main__":

    class fullTimeEmployee(employee):
        def __init__(self, family, salary, department):
            employee.__init__(self, family, salary, department)

# instances of Fulltime Employee class and Employee class and call their member functions
if __name__ == '__main__':
    e = employee("David", 10000, "HR")
    print(e.count)
    g = employee("Jerry", 12000, "intern")
    print(g.getInfor())
    s = fullTimeEmployee("Tod",20000, "R&D")

    print(employee.count)


# Question 2
import numpy as np
Z = np.random.random((10,10))
result1 = [min(row) for row in Z]
result2 = [max(row) for row in Z]
print(result1, result2)