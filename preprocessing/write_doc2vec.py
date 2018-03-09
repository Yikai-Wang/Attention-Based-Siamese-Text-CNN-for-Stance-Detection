#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 16:45:44 2018

@author: Hendry
"""

import gensim.models as g
import codecs

#parameters

model="./apnews_dbow/doc2vec.bin"
test_docs="./test_titles.txt"
output_file="./testTitle_vectors.txt"

#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000

#load model
m = g.Doc2Vec.load(model)
test_docs = [ x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines() ]

#infer test vectors
output = open(output_file, "w")
for d in test_docs:
    output.write( " ".join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n" )
output.flush()
output.close()

