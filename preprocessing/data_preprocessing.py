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
#testRes = splitData(test_stances,2)


model = gensim.models.KeyedVectors.load_word2vec_format('./vectors100.bin', binary=True)

def dataVectorlize(data,model):
    vec = np.zeros([len(data),100])
    for i in range(len(data)):
        k=0
        for j in range(len(data[i])):
            try:
                vec[i]+=model.word_vec(data[i][j].lower())
                k+=1
            except KeyError:k-=1
        vec[i]/=max(k,1)
    return vec
trainDocsVec = dataVectorlize(trainDocs,model)
trainTitleVec = dataVectorlize(trainTitle,model)
testDocsVec = dataVectorlize(testDocs,model)
testTitleVec = dataVectorlize(testTitle,model)
trainTitleDocsVec = np.zeros_like(trainTitleVec)
testTitleDocsVec = np.zeros_like(testTitleVec)
for i in range(len(trainTitle)):
    idx = np.where(trainDocsIdx==trainTitleIdx[i])
    trainTitleDocsVec[i]=trainDocsVec[idx]
for i in range(len(testTitle)):
    idx = np.where(testDocsIdx==testTitleIdx[i])
    testTitleDocsVec[i]=testDocsVec[idx]
    
#x1Train = trainTitleDocsVec[:round(0.9*len(trainTitleDocsVec)),:]
#x2Train = trainTitleVec[:round(0.9*len(trainTitleDocsVec)),:]
#yTrain = trainRes[:round(0.9*len(trainTitleDocsVec))]
#x1Valid = trainTitleDocsVec[round(0.9*len(trainTitleDocsVec)):,:]
#x2Valid = trainTitleVec[round(0.9*len(trainTitleDocsVec)):,:]
#yValid = trainRes[round(0.9*len(trainTitleDocsVec)):]
uniIdx = np.unique(trainTitleIdx)
uniIdxTest = uniIdx[round(0.95*len(uniIdx)):]
validIdx  = np.argwhere(trainTitleIdx == uniIdxTest[0])

for i in range(len(uniIdxTest)-1):
    validIdx = np.append(validIdx,np.argwhere(trainTitleIdx == uniIdxTest[i+1]))
validIdx = sorted(validIdx)
fullIdx = list(range(len(trainTitleIdx)))
trainIdx =  list(set(fullIdx).difference(set(validIdx)))
x1Train = trainTitleDocsVec[trainIdx,:]
x2Train = trainTitleVec[trainIdx,:]
trainRes = np.array(trainRes)
yTrain = trainRes[trainIdx]
x1Valid = trainTitleDocsVec[validIdx,:]
x2Valid = trainTitleVec[validIdx,:]
yValid = trainRes[validIdx]


