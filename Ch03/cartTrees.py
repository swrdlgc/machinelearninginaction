'''
Created on Sep 16, 2014
Decision Tree Source Code(cart) for Machine Learning in Action Ch. 3
@author: swrd
'''
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
    sltflag = []
    maxsltflag = pow(2, len(labels))
    for i in range(maxsltflag):
        if (maxsltflag - 1 - i) not in sltflag:
            sltflag.append(i)
            subslt = []
            for j in range(len(labels)):
                if ((1<<j)&i) != 0:
                    subslt.append(labels[j])
            splits.append(subslt)
    if len(labels) <= 1: return []
    return splits[1:len(splits)]

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

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    bestGini = float('inf'); 
    bestFeature = -1
    featSet = []
    for i in range(numFeatures):        #iterate over all the features
        splits = getSplits(dataSet, i)
        for value in splits:            #iterate over all sub feature set
            yesDataSet, noDataSet = binSplitDataSet(dataSet, i, value)
            gini = getGini(yesDataSet)*len(yesDataSet)/len(dataSet) + getGini(noDataSet)*len(noDataSet)/len(dataSet)
            #print gini, i, value
            if(bestGini > gini):
                bestGini = gini
                bestFeature = i
                featSet = value
    return bestFeature, featSet

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
    bestFeat, value = chooseBestFeatureToSplit(dataSet)
    #print bestFeat, value
    #print labels
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    yesDataSet, noDataSet = binSplitDataSet(dataSet, bestFeat, value)
    allvalue = list(set([vec[bestFeat] for vec in dataSet ]))
    rightvalue = [key for key in allvalue if key not in value]
    left = ",".join(value)
    right = ",".join(rightvalue)
    myTree[bestFeatLabel][left] = createTree(yesDataSet, list(labels))
    myTree[bestFeatLabel][right] = createTree(noDataSet, list(labels))
    return myTree                            
    
#modify to here
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
    
