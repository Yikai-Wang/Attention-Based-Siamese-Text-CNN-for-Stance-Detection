import numpy as np
import random

def normalizeDoc(Docs,lengthDoc,defaultWord):
	for i in range(Doc):
		if len(Doc[i])>lengthDoc:
			randStart = random.randint(0,len(Doc[i])-lengthDoc-1)
			Doc[i] = Doc[i][randStart:randStart+lengthDoc]
		elif len(Doc[i])<lengthDoc:
			Doc[i] += [defaultWord for j in range(lengthDoc-len(Doc[i]))]
	return Docs

def batch_iter(data, batch_size, num_epochs, shuffle=True):

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            # print('epoch = %d,batch_num = %d,start = %d,end_idx = %d' % (epoch,batch_num,start_index,end_index))
            yield shuffled_data[start_index:end_index]