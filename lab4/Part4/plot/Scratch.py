
import matplotlib.pyplot as plt 

def read_in_file(file):
    data = []
    with open(file, 'r') as f:
        next(f)
        for line in f:
            l = line.split(',')
            data.append(float(l[2]))
    return data



#Data = read_in_file('1stRun.csv')

#Data = read_in_file('2ndRun.csv')

Data = read_in_file('Part4/plot/1532465794_cross_entropy.csv')
steps = []
for x in range (0,len(Data)):
    steps.append(x* 10)


#plt.plot(steps, Data, label="Adam")
#plt.plot(steps, Data, label="RMSProp", color="red")
plt.plot(steps, Data, label="Accuracy ", color='green')
plt.ylabel("Accuracy")
plt.xlabel("Steps")
plt.legend()
plt.show()