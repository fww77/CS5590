#%%
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


#colors for marking thd data on the plot. red will be used to mark the wrong predictions
colors = ['navy', 'turquoise', 'orange']
# used to plot all results
def plot_results(prediction,data,test, xlabel:str ,ylabel: str,title: str, index: int):
    #figure auto increases
    plt.figure()
    for i, p, t in zip(range(len(test)),prediction, test):
        if p != t:
            color = 'red'
        else:
            color = colors[t]
        plt.scatter(x_test[i][index], x_test[i][index + 1], c=color)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)  



lr = LogisticRegression()
lda = LinearDiscriminantAnalysis(n_components=4)

data = load_iris()
#separate the data and the target
iris_x = data.data
iris_y = data.target

#splits the data into test and train sets. set to  50/50 to get more incorrect results
x_train, x_test, y_train, y_test = train_test_split(iris_x,iris_y, test_size=0.5)

#train the data
lda_train = lda.fit(x_train,y_train)
lr.fit(x_train, y_train)

#array of the predict results
lda_prediction = lda.predict(x_test)
lr_prediction = lr.predict(x_test)


#sets the labels
xlabel = "Sepal Length (cm)"
ylabel = "Sepal Width (cm)"
title = 'LDA of IRIS dataset "Sepal"'
#plots LDA for sepal
plot_results(lda_prediction,x_test,y_test,xlabel,ylabel,title,0)

#sets the labels
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title('LDA of IRIS dataset "Petals"') 
#plots LDA for petals   
plot_results(lda_prediction,x_test,y_test,xlabel,ylabel,title,2)


#sets the labels
xlabel = "Sepal Length (cm)"
ylabel = "Sepal Width (cm)"
title = 'LR of IRIS dataset "Sepal"' 
#plots LR for sepal
plot_results(lr_prediction,x_test,y_test,xlabel,ylabel,title,0)
    


#sets the label
xlabel = "Petal Length (cm)"
ylabel = "Petal Width (cm)"
title ='LR of IRIS dataset "Petals"'
#plots LR for petal
plot_results(lr_prediction,x_test,y_test,xlabel,ylabel,title,2)

plt.show()

