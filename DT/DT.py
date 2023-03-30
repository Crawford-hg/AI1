import pandas as pd
import numpy as np
import sys


class node:
    def __init__(self) -> None:
        pass


class leaf(node):
    def __init__(self, classificatiom, probability):
        self.classificatiom = classificatiom
        self.probability = probability

    def getNodeName(self):
        return self.classificatiom


class branch(node):
    def __init__(self, attributes, bestAttribute, left, right):
        self.attributes = attributes
        self.bestAttribute = bestAttribute
        self.left = left
        self.right = right

    def getNodeName(self):
        return self.bestAttribute


class instance:
    def __init__(self, attribute, catagory):
        self.values = []
        self.catagory = ""
        self.catagory = catagory
        for i in range(1, len(attribute)):
            self.values.append(attribute[i])

    def getCatagory(self):
        return self.catagory

    def getValues(self, i):
        return self.values[i]

    def printAll(self):
        print(self.values)

    def valSize(self):
        return len(self.values)


class DT:
    def __init__(self) -> None:
        self.mostCommon = []
        self.allAttributes = []
        self.allInstances = []
        self.root = node()
        self.correctPredictions = 0
        self.totalPredictions = 0
        self.testInstances =[]

    def parseFile(self, filename):
        data = []
        with open(filename) as f:
            for line in f:
                data.append(line.split())
        self.allAttributes = data[0]

        for i in range(1, len(data)):
            self.allInstances.append(instance(data[i], data[i][0]))
        self.allAttributes.remove("Class")
    
    def parseTestFile(self, filename):
        data = []
        with open(filename) as f:
            for line in f:
                data.append(line.split())

        for i in range(1, len(data)):
            self.testInstances.append(instance(data[i], data[i][0]))

    def setRoot(self):
        self.root = self.buildTree(self.allInstances, self.allAttributes)

    def buildTree(self, instances, attributes):
        if len(instances) == 0:
            return leaf(
                self.getMostCommon(self.allInstances),
                self.getProbability(self.allInstances),
            )
        elif self.pureInstance(instances):
            return leaf(instances[0].getCatagory(), 1)
        elif not attributes:
            return leaf(self.getMostCommon(instances), self.getProbability(instances))
        else:
            return self.createBranch(instances, attributes)

    def createBranch(self, instances, attributes):
        bestPurity = 1
        bestTrueInstances = []
        bestFalseInstances = []
        bestAttribute = ""
        for a in range(len(self.allAttributes)):
            if self.allAttributes[a] not in attributes:
                continue
            trueInstaneces = []
            falseInstances = []
            for i in range(len(instances)):
                currentInstance = instances[i]

                if currentInstance.getValues(a) == "true":
                    trueInstaneces.append(currentInstance)
                else:
                    falseInstances.append(currentInstance)
            weightedImpurity = self.weightedImpurity(trueInstaneces, falseInstances)

            if weightedImpurity < bestPurity:
                bestPurity = weightedImpurity
                bestTrueInstances = trueInstaneces
                bestFalseInstances = falseInstances
                bestAttribute = self.allAttributes[a]

        attCopy = attributes.copy()
        if bestAttribute in attCopy:
            attCopy.remove(bestAttribute)

        return branch(
            attCopy,
            bestAttribute,
            self.buildTree(bestTrueInstances, attCopy.copy()),
            self.buildTree(bestFalseInstances, attCopy.copy()),
        )

    def weightedImpurity(self, instances1, instances2):
        imp1 = self.impurity(instances1)
        imp2 = self.impurity(instances2)
        divisor = len(instances1) + len(instances2)
        return (
            (len(instances1) / divisor) * imp1 + (len(instances2) / divisor) * imp2
        ) / 2

    def impurity(self, instances):
        if len(instances) == 0:
            return 0
        count = 0
        for i in instances:
            if i.getCatagory() == "live":
                count += 1

        return (count * (len(instances) - count)) / len(instances) ** 2

    def pureInstance(self, instances):
        catagory = instances[0].getCatagory()
        for i in instances:
            if i.getCatagory() != catagory:
                return False
        return True

    def getProbability(self, instances):
        if len(instances) == 0:
            return 0
        count = 0
        for i in instances:
            if i.getCatagory() == "live":
                count += 1
        return count / len(instances)

    def getMostCommon(self, instances):
        countLive = 0
        countDie = 0
        for i in instances:
            if i.getCatagory() == "live":
                countLive += 1
            else:
                countDie += 1
        if countLive > countDie:
            return "live"
        else:
            return "die"

    def printTree(self):
        self.printTreeHelper(self.root, 0)

    def printTreeHelper(self, node, depth):
        str = "\t" * depth
        if isinstance(node, leaf):
            print(str, node.getNodeName(),'Probability = ', node.probability)
        if isinstance(node, branch):
            print(str, node.getNodeName(), "True")
            self.printTreeHelper(node.left, depth + 1)
            print(str, node.getNodeName(), "False")
            self.printTreeHelper(node.right, depth + 1)

    def testTree(self):
        for i in self.testInstances:
            self.testTreeHelper(self.root, i)
        print('Baseline', self.getProbability(self.allInstances))
        print('Correct', self.correctPredictions , '/' , self.totalPredictions)
        print('Accuracy',self.correctPredictions/self.totalPredictions)
    
    def testTreeHelper(self, node, instance):
        if isinstance(node, leaf):
            self.totalPredictions += 1
            if node.getNodeName() == instance.getCatagory():
                self.correctPredictions += 1
        if isinstance(node, branch):
            if instance.getValues(self.allAttributes.index(node.getNodeName())) == "true":
                self.testTreeHelper(node.left, instance)
            else:
                self.testTreeHelper(node.right, instance)
trainingPath = './'+sys.argv[1]
testPath = './'+sys.argv[2]
tree = DT()
tree.parseFile(trainingPath)
tree.setRoot()
tree.printTree()
tree.parseTestFile(testPath)
tree.testTree()
