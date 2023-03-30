import math
import pandas as pd
import numpy as np
import sys
class KNN:
    def __init__(self,trainPath,testPath,k):
        self.k = k
        self.trainingDF = pd.read_csv(trainPath,sep=' ')
        self.testDF = pd.read_csv(testPath,sep=' ')
        self.trainingClass = self.trainingDF["Class"]
        self.testClass = self.testDF["Class"]
        self.testDF.drop("Class", axis='columns')
        self.trainingDF.drop("Class", axis='columns')

        self.normalizedTraining = self.trainingDF.copy()
        self.normalizedTest = self.testDF.copy()
        self.calculatedClass = []



    

    #initialses the variables for the kNN to run
    def initialise(self):
        for column in self.trainingDF.columns:
            train = self.trainingDF[column]
            test = self.testDF[column]
            self.normalizedTraining[column] = (train) / (train.max() - train.min())
            self.normalizedTest[column] = (test) / (train.max() - train.min())

        self.normalizedTraining = self.normalizedTraining.to_numpy()
        self.normalizedTest = self.normalizedTest.to_numpy()

    #Method to calculate the distance between two wines
    def calculateDistance(self,trainingWine,testWine):
        distance = 0
        for i in range(0,13):
            distance += (testWine[i] - trainingWine[i])**2
        return distance

    #Method to run the kNN algorithm
    def doKNN(self):
        for i in range(0,len(self.normalizedTest)):
            distances=[]
            minDistance = []
            closestNeighbor = float("inf")
            closestIndex = 0
            minIndex = []

            for j in range(0,len(self.normalizedTraining)):
                distance = self.calculateDistance(self.normalizedTraining[j],self.normalizedTest[i])
                distances.append(distance)

                if distance<closestNeighbor:
                    closestNeighbor = distance
                    closestIndex = j

            sorted = distances.copy()
            sorted.sort()

            for i in range(0,self.k):
                minDistance.append(sorted[i])
                minIndex.append(distances.index(sorted[i]))
        
            classes=[]
            for i in range(0,len(minIndex)):
                classes.append(self.trainingClass[minIndex[i]])
            self.calculatedClass.append(max(set(classes), key = classes.count))
            print(classes[0:10])
            
            
        correct =0
        for i in range(0,len(self.calculatedClass)):
            if self.calculatedClass[i] == self.testClass[i]:
                correct += 1
        print(correct/len(self.calculatedClass)*100)

trainingPath = './'+sys.argv[1]
testPath = './'+sys.argv[2]
kVal = int(sys.argv[3])
knn = KNN(trainingPath,testPath,kVal)
knn.initialise()
knn.doKNN()

        

        
            



        



   

