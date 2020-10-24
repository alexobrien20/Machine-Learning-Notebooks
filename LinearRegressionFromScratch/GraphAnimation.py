import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("RegressionData.csv")

X = df['X'][:100]
Y = df['Real Y'][:100]
h1, = plt.plot(X,Y,'o')
x_values = [0,20,40,60,80,100]
plt.ion()
for i in range(998):
    y = df.iloc[i+2][3:]
    plt.ylim(-5,65)
    plt.xticks(x_values)
    plt.xlabel("X")
    plt.title("Graph of Y Versus X with our 'learned' line of best fit")
    plt.ylabel("Y")
    plt.plot(y,color='orange',label='predicted')
    plt.plot(Y,'bo',label='actual')
    plt.legend(loc="upper left")
    plt.draw()
    plt.pause(0.05)
    plt.clf()
