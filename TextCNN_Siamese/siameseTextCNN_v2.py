#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 23:56:18 2018

@author: Hendry
"""

import tensorflow as tf
import numpy as np

class siameseTextCNN(object):

    # Create model
    def __init__(self,w2v_model, seqLengthDoc, seqLengthTitle, vocabSize,
      embeddingSize, filterSizes, numFilters,numClasses=4, numHidden=10, l2_reg_lambda=0.01):
        self.x1 = tf.placeholder(tf.int32, [None, seqLengthDoc],name="input_x1")
        self.x2 = tf.placeholder(tf.int32, [None, seqLengthTitle],name="input_x2")
        self.y = tf.placeholder(tf.float32, [None, numClasses],name="input_y")
        self.y0 = self.y[:,0]
        self.y1 = self.y[:,1:]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_reg = tf.constant(0.0)
        maxLenX1 = seqLengthDoc
        maxLenX2 = seqLengthTitle
        if w2v_model is None:
            self.W = tf.Variable(
                tf.random_uniform([vocabSize, embeddingSize], -1.0, 1.0),
                name="word_embeddings")
        else:
            self.W = tf.get_variable("word_embeddings",initializer=w2v_model.vectors.astype(np.float32))
            self.embeddedChars1 = tf.expand_dims(tf.nn.embedding_lookup(self.W, self.x1), -1)
            self.embeddedChars2 = tf.expand_dims(tf.nn.embedding_lookup(self.W, self.x2), -1)
        print(self.embeddedChars2)
        output1 = []
        output2 = []
        numFiltersTotal = numFilters * len(filterSizes)
        # Construct Filters
        for i, filterSize in enumerate(filterSizes):
            filterShape = [filterSize, embeddingSize, 1, numFilters]
            for k in [1,2]:
                with tf.name_scope("Conv-Maxpool-Layer-%s-%s" % (str(k),filterSize)):
                    # Convolution Layer
                    W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(1.0, shape=[numFilters]), name="b")
                    
                    conv = tf.nn.conv2d(eval('self.embeddedChars'+str(k)),W,
                        strides=[1, 1, 1, 1],padding="VALID",name="conv")
                    # Activate function
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h, ksize=[1, eval('maxLenX'+str(k)) - filterSize + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    eval('output'+str(k)+'.append(pooled)') 


        self.hiddenPooled1 = tf.reshape(tf.concat( output1,3), [-1, numFiltersTotal], name='hiddenPooled1')
        self.hiddenPooled2 = tf.reshape(tf.concat( output2,3), [-1, numFiltersTotal], name='hiddenPooled2')
        

        # Compute similarity
        with tf.name_scope("similarity"):
            W = tf.get_variable(
                "W",
                shape=[numFiltersTotal, numFiltersTotal],
                initializer=tf.contrib.layers.xavier_initializer())
            self.transform1 = tf.matmul(self.hiddenPooled1, W)
            self.sims = tf.reduce_sum(tf.multiply(self.transform1, self.hiddenPooled2), 1, keep_dims=True)
            print(self.sims)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Make input for classification
        self.Input = tf.concat([self.hiddenPooled1, self.sims, self.hiddenPooled2],1, name='Input')

        # hidden layer
        with tf.name_scope("hidden"):
            W = tf.get_variable(
                "W_hidden",
                shape=[2*numFiltersTotal+1, numHidden],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[numHidden]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.hiddenOutput = tf.nn.relu(tf.nn.xw_plus_b(self.Input, W, b, name="hiddenOutput"))

        # Add dropout
        with tf.name_scope("dropout"):
            self.hDrop = tf.nn.dropout(self.hiddenOutput, self.dropout_keep_prob, name="hidden_output_drop")


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[numHidden, 4],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[4]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.hDrop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.scores0 = self.scores[:,0]
            self.scores1 = self.scores[:,1:]




        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            
            
            losses1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores0, labels=self.y0)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores1, labels=self.y1)
            self.loss = 0.75*tf.reduce_mean(losses2)+0.25*tf.reduce_mean(losses1) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")





