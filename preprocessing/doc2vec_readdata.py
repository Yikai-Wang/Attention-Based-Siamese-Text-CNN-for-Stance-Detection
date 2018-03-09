#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 23:48:34 2018

@author: Hendry
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

trainDocsVec = np.loadtxt('trainDocs_vectors.txt')
trainTitleVec = np.loadtxt('trainTitle_vectors.txt')
trainTitleDocsVec = np.zeros_like(trainTitleVec)
for i in range(len(trainTitle)):
    idx = np.where(trainDocsIdx==trainTitleIdx[i])
    trainTitleDocsVec[i]=trainDocsVec[idx]
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


