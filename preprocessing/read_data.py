import csv
def readRawData(name):	
	csv_reader = list(csv.reader(open('./fnc_data/'+name, encoding='utf-8')))
	csv_reader.remove(csv_reader[0])
	return csv_reader

def splitData(data,num):
	dataEdit = []
	for i in range(len(data)):
		dataEdit.append(data[i][num])
	return dataEdit




if __name__=='__main__':
	train_bodies = readRawData('train_bodies.csv')
	trainDocs = splitData(train_bodies,1)
	train_stances = readRawData('train_stances.csv')
	test_bodies = readRawData('test_bodies.csv')
	test_stances_unlabeled = readRawData('test_stances_unlabeled.csv')

	
