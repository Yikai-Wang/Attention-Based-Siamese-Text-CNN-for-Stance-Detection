# Fake News Challenge
Here is the final project for NLP course.

Our work include several parts:

1. Data preprocessing
2. Conventional Machine Learning Method
3. Seq2seq Attention model
4. TextCNN and Siamese Network
5. Others (e.g. Related work in the competition, future work)

### 1. Data preprocessing

We provide several data preprocessing methods: BoW (Bag of words), TF-IDF, word2vec, doc2vec.

Each py file produce  x_1 (doc representation) x_2 (title representation) and y (label). 

These data can output as spy-data, which can be used in the model.

### 2. Conventional Machine Learning
We provide py file to classify the instances by Conventional Machine Learning (e.g. SVM, Random Forest), the codes are implemented on sklearn.
Environments requirements:
sklearn
numpy


### 3. Seq2seq Attention model

The codes is generally base on one attention based sequence-to-sequence model (https://github.com/abisee/pointer-generator) with his pretrained model. To use the code to generate summary of the text. Run:
python3 run_summarization.py --mode=decode --data_path=/path/to/chunked/val_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
Environments requirements:
tensorflow (1.2.1)
numpy

### 4. TextCNN and Siamese Network

The codes of TextCNN with Siamese Network are our novel work, the construction of CNN refered conventional TextCNN (https://github.com/dennybritz/cnn-text-classification-tf), we extend the Siamese structure for our purpose.

Main_v5 is the newest version till now.

Before using, you need to add raw data into fnc_data folder, and implement word2vec bin file named vector100.bin in the root folder.

Moreover, tensorflow (1.3.0 & 1.4.1 are supported, but either under 1.0 or 1.6 causes some minor conflicts), word2vec and numpy package.

''*_FineGrain": means 3-class fine grain classification (Specific relation of "related")

"*_FullDoc" : means use full documents word2vec matrix as x_1

Default: classification of related/unrelated, use attention based title to replace full documents.

### 5. Others
The detail step of UCL's work are in folder: ./UCL_Repeat/README.md
Environments requirements:
Python 3.5
Tensorflow 0.12.1
Scipy
Numpy

