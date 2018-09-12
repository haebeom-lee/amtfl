from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def mnist_imb(path, batch_size, nlist):
    mnist = input_data.read_data_sets(path, one_hot=True, validation_size=0)
    xtrva, ytrva = mnist.train.images, mnist.train.labels
    xte, yte = mnist.test.images, mnist.test.labels

    # train
    ytrva_ = np.argmax(ytrva, 1)
    xtr_list = [xtrva[ytrva_==k][:nlist[k],:] for k in range(10)]
    ytr_list = [ytrva[ytrva_==k][:nlist[k],:] for k in range(10)]

    # validation
    xva_list = [xtrva[ytrva_==k][nlist[k]:,:] for k in range(10)]
    yva_list = [ytrva[ytrva_==k][nlist[k]:,:] for k in range(10)]

    xtr, ytr = np.concatenate(xtr_list, axis=0), np.concatenate(ytr_list, axis=0)
    xva, yva = np.concatenate(xva_list, axis=0), np.concatenate(yva_list, axis=0)

    n_train_batches = 1100/batch_size
    n_val_batches = 58900/batch_size
    n_test_batches = 10000/batch_size
    return xtr, ytr, xva, yva, xte, yte, n_train_batches, n_val_batches, n_test_batches
