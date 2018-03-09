#! /usr/bin/env python
#coding=utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from siameseTextCNN_v3 import *
import math
from tensorflow.contrib import learn
from util import *
from cnn_loaddata_v2 import *
import word2vec
## Parameters

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("w2v_file", "./vectors100.bin", "w2v_file path")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")




def train(w2v_model):
    # Training
    # ==================================================
    x1Train, x1Dev, x2Train, x2Dev, yTrain, yDev, vocabSize = loaddata(w2v_model,1,0)
    
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = siameseTextCNN(
                w2v_model, seqLengthDoc=x1Train.shape[1], seqLengthTitle = x2Train.shape[1],
                vocabSize=vocabSize, embeddingSize=FLAGS.embedding_dim,
                filterSizes=list(map(int, FLAGS.filter_sizes.split(","))),
                 numFilters=FLAGS.num_filters,numClasses=2, l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
#            grad_summaries = []
#            for g, v in grads_and_vars:
#                if g is not None:
#                    
#                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':','_')), g)
#                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':','_')), tf.nn.zero_fraction(g))
#                    grad_summaries.append(grad_hist_summary)
#                    grad_summaries.append(sparsity_summary)
#            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
#            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x1Batch, x2Batch, yBatch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.x1: x1Batch,
                  cnn.x2: x2Batch,
                  cnn.y: yBatch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                # _, step, summaries, loss, accuracy,(w,idx) = sess.run(
                #     [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy,cnn.get_w2v_W()],
                #     feed_dict)
                _, step , loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # print w[:2],idx[:2]
                


            def dev_step(x1Batch, x2Batch, yBatch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.x1: x1Batch,
                  cnn.x2: x2Batch,
                  cnn.y: yBatch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, loss, accuracy, sims, pres  = sess.run(
                    [global_step, cnn.loss, cnn.accuracy, cnn.sims, cnn.scores],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                y0 = np.argmax(yBatch,1)
                ypred = np.argmax(pres,1)
                acc0 = sum(ypred==y0)/len(y0)
                if len(y0[y0>0])>0:
                    acc1 = sum(np.argmax(pres,1)[y0>0]==y0[y0>0])/len(y0[y0>0])
                else:
                    acc1 = 1
                    print('Warning: all titles in the batch are unrelated')
                print('0.75*Acc1+0.25*Acc0 = ',acc0/4+acc1*3/4,'\n')
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                


            # Generate batches
            print("Generate batches\n")
            batches = batch_iter(
                list(zip(x1Train,x2Train,yTrain)), FLAGS.batch_size, FLAGS.num_epochs)

            def dev_test():
                batches_dev = batch_iter(list(zip(x1Dev, x2Dev, yDev)), FLAGS.batch_size, 1)
                for batch_dev in batches_dev:
                    x1BatchDev, x2BatchDev, yBatchDev = zip(*batch_dev)
                    dev_step(x1BatchDev, x2BatchDev, yBatchDev, writer=dev_summary_writer)

            # Training loop. For each batch...
            for batch in batches:
                x1BatchTrain, x2BatchTrain, yBatchTrain = zip(*batch)
                train_step(x1BatchTrain, x2BatchTrain, yBatchTrain )
                current_step = tf.train.global_step(sess, global_step)
                # Training loop. For each batch...
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_test()


                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__":  
    w2v_model = word2vec.load('./vectors100.bin')
    train(w2v_model)
