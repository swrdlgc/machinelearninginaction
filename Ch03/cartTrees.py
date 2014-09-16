'''
Created on Sep 16, 2014
Decision Tree Source Code(cart) for Machine Learning in Action Ch. 3
@author: swrd
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def getGini(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    gini = 1.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        gini -= prob * prob
    return gini

def getSplits(dataSet, axis):
    labels = list(set(zip(*dataSet)[axis]))
    splits = []
    for i in range(pow(2, len(labels))):
        subslt = []
        for j in range(len(labels)):
            if ((1<<j)&i)!=0:
                subslt.append(labels[j])
        splits.append(subslt)
    if len(splits) <= 2: return []
    return splits[1:len(splits)-1]

def binSplitDataSet(dataSet, axis, value):
    yesDataSet = []
    noDataSet = []
    for featVec in dataSet:
        reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
        reducedFeatVec.extend(featVec[axis+1:])
        if featVec[axis] in value:
            yesDataSet.append(reducedFeatVec)
        else:
            noDataSet.append(reducedFeatVec)
    return yesDataSet, noDataSet

#modify to here
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        subDataSetArr = []
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            subDataSetArr.append(subDataSet)
        #infoGain = getInfoGain(baseEntropy, subDataSetArr)	# C3.0
        infoGain = getInfoGainRatio(getInfoGain(baseEntropy, subDataSetArr), subDataSetArr) # C4.5
        #print infoGain
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
