import trees
import treePlotter

def testDataSet(file):
	fr=open(file)
	dataSet = [inst.strip().split('\t') for inst in fr.readlines()]
	dataSetLabels = dataSet[0]
	dataSet = dataSet[1:]
	return trees.createTree(dataSet, dataSetLabels)