#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:05:24 2018

@author: Hendry
"""

from read_data import *
from TokenizeSentences import *
import gensim
import numpy as np
def onehot(data,nClass):
	data2 = np.zeros([len(data),nClass])
	for i in range(nClass):
		data2[np.where(data==i),i]= 1
	return data2


def get_text_idx(text,vocab,max_document_length):
    text_array = np.zeros([len(text), max_document_length],dtype=np.int32)

    for i,x in  enumerate(text):
        words = x
        for j, w in enumerate(words):
            if w in vocab:
                text_array[i, j] = vocab[w]
            else :
                text_array[i, j] = vocab['the']

    return text_array

def loaddata(w2v_model):
	train_bodies = readRawData('train_bodies.csv')
	trainDocs = TokenizeSentences(splitData(train_bodies,1))
	trainDocsIdx = np.array(splitData(train_bodies,0)).astype('int')
	train_stances = readRawData('train_stances.csv')
	trainTitle = TokenizeSentences(splitData(train_stances,0))
	trainTitleIdx = np.array(splitData(train_stances,1)).astype('int')
	trainRes = np.array(splitData(train_stances,2))
	trainRes[np.where(trainRes=='unrelated')]='0'
	trainRes[np.where(trainRes=='agree')]='1'
	trainRes[np.where(trainRes=='disagree')]='2'
	trainRes[np.where(trainRes=='discuss')]='3'
	trainRes =trainRes.astype('int')
	maxDocLength = 0
	for i in range(len(trainDocs)):
		maxDocLength = max(maxDocLength,len(trainDocs[i]))
	maxTitleLength = 0
	for i in range(len(trainTitle)):
		maxTitleLength = max(maxTitleLength,len(trainTitle[i]))
	trainDocs = get_text_idx(trainDocs,w2v_model.vocab_hash,maxDocLength)
	trainTitle = get_text_idx(trainTitle,w2v_model.vocab_hash,maxTitleLength)
	trainTitleDocs = [[] for i in range(len(trainTitle))]
	for i in range(len(trainTitle)):
	    idx = np.where(trainDocsIdx==trainTitleIdx[i])
	    trainTitleDocs[i]=trainDocs[int(idx[0])]
	trainTitleDocs = np.array(trainTitleDocs)
	trainDocs = np.array(trainDocs)
	trainTitle = np.array(trainTitle)
	uniIdx = np.unique(trainTitleIdx)
	uniIdxTest = uniIdx[round(0.95*len(uniIdx)):]
	validIdx  = np.argwhere(trainTitleIdx == uniIdxTest[0])
	for i in range(len(uniIdxTest)-1):
	    validIdx = np.append(validIdx,np.argwhere(trainTitleIdx == uniIdxTest[i+1]))
	validIdx = sorted(validIdx)
	fullIdx = list(range(len(trainTitleIdx)))
	trainIdx =  list(set(fullIdx).difference(set(validIdx)))
	x1Train = trainTitleDocs[trainIdx]
	x2Train = trainTitle[trainIdx]
	trainRes = np.array(trainRes)
	y0Train = trainRes[trainIdx]
	x1Valid = trainTitleDocs[validIdx]
	x2Valid = trainTitle[validIdx]
	y0Valid = trainRes[validIdx]
	yValid = onehot(y0Valid,4)
	yTrain = onehot(y0Train,4)
	vocab_size = len(w2v_model.vocab_hash)
	return x1Train, x1Valid, x2Train, x2Valid, yTrain, yValid, vocab_size
	
	
	
	
	
	
	
	
	
	
	