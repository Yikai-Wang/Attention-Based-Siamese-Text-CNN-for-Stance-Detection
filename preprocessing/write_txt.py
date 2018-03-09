# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from read_data import *
from TokenizeSentences import *
import gensim
import numpy as np


train_bodies = readRawData('train_bodies.csv')
trainDocs = TokenizeSentences(splitData(train_bodies,1))
trainDocsIdx = np.array(splitData(train_bodies,0)).astype('int')
train_stances = readRawData('train_stances.csv')
trainTitle = TokenizeSentences(splitData(train_stances,0))
trainTitleIdx = np.array(splitData(train_stances,1)).astype('int')
trainRes = splitData(train_stances,2)
test_bodies = readRawData('test_bodies.csv')
testDocs = TokenizeSentences(splitData(test_bodies,1))
testDocsIdx = np.array(splitData(test_bodies,0)).astype('int')
test_stances = readRawData('test_stances_unlabeled.csv')
testTitle = TokenizeSentences(splitData(test_stances,0))
testTitleIdx = np.array(splitData(test_stances,1)).astype('int')

def remove_n(var):
	for i in var:
		while '\n' in i:i.remove('\n')
	return var
trainDocs = remove_n(trainDocs)
testDocs = remove_n(testDocs)
trainTitle = remove_n(trainTitle)
testTitle = remove_n(testTitle)

def write_txt(var,file):
	f = open(file,'w')
	for i in range(len(var)):
		f.write(' '.join(var[i])+'\n')
	f.close()
write_txt(trainDocs,'./train_documents.txt')
write_txt(testDocs,'./test_documents.txt')
write_txt(trainTitle,'./train_titles.txt')
write_txt(testTitle,'./test_titles.txt')

