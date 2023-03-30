import pandas as pd
import numpy as np
import sys

class Perceptron:
    def __init__(self,path):
        self.df = pd.read_csv(path,sep=' ')
        self.bias = 1
        
        self.classData = self.df["class"]
        self.df = self.df.drop("class",axis=1)
        print(self.df)
        self.weights = np.ones(len(self.df.columns))
        self.epoch = 0        
        self.allAcc = []

    def activation(self):
        while self.epoch<200 and self.checkAccuracy() != 1:
            for i in range(len(self.df.values)):
                pred = 1 if np.dot(self.df.values[i], self.weights) + self.bias > 0 else 0
                actual = 1 if self.classData[i] == 'g' else 0
        
                if pred != actual:
                    for value in range(len(self.weights)):
                        self.weights[value] += self.df.values[i][value] * (actual - pred)
                    self.bias += (actual - pred)

            
            currentAccuracy = self.checkAccuracy()
            self.allAcc.append(currentAccuracy)
            print(currentAccuracy)
            if currentAccuracy != 1 and self.epoch<200:
                self.epoch+=1
        print("Epochs: ",self.epoch)
        print("Average Accuracy: ", np.mean(self.allAcc))
        print("Highest Accuracy: ", max(self.allAcc))
        print("Lowest Accuracy: ", min(self.allAcc))

    def checkAccuracy(self):
        count = 0
        for i in range(len(self.df)):
            pred = 1 if np.dot(self.df.values[i], self.weights) + self.bias > 0 else 0
            actual = 1 if self.classData[i] == 'g' else 0
            if pred == actual:
                count+=1
        return count / len(self.df)


path = './'+sys.argv[1]
print(path)
test = Perceptron(path)
test.activation()
