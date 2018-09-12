from __future__ import print_function
import tensorflow as tf
import numpy as np
from amtfl import amtfl
from accumulator import Accumulator
from mnist import mnist_imb
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mnist_path', type=str, default='./mnist')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--savedir', type=str, default=None)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.01)
args = parser.parse_args()

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)

savedir = './results/amtfl' if args.savedir is None else args.savedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)

batch_size = args.batch_size
nlist = [20*(10-i) for i in range(10)]
xtr, ytr, xva, yva, xte, yte, n_train_batches, \
        n_val_batches, n_test_batches = mnist_imb(args.mnist_path, batch_size, nlist)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
net = amtfl(x, y, True, nlist, args.alpha, args.gamma)
tnet = amtfl(x, y, False, nlist, args.alpha, args.gamma, reuse=True)

def run():
    loss = net['cent_loss'] + net['recon_loss'] + net['l1_decay'] + net['l2_decay']
    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
            [n_train_batches*args.n_epochs/2], [1e-4, 1e-5])
    train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(net['weights'])
    logfile = open(os.path.join(savedir, 'train.log'), 'w', 0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_logger = Accumulator('cent_loss', 'recon_loss',
            'l1_decay', 'l2_decay', 'acc')
    train_to_run = [train_op, net['cent_loss'], net['recon_loss'],
            net['l1_decay'], net['l2_decay'], net['acc']]
    val_logger = Accumulator('cent_loss', 'acc')
    val_to_run = [tnet['cent_loss'], tnet['acc']]

    for i in range(args.n_epochs):
        xytr = np.concatenate((xtr, ytr), axis=1)
        np.random.shuffle(xytr)
        xtr_, ytr_ = xytr[:,:784], xytr[:,784:]

        if (i+1) % 100 == 0:
            line = 'Epoch %d start, learning rate %f' % (i+1, sess.run(lr))
            print (line)
            logfile.write(line + '\n')

        train_logger.clear()
        start = time.time()

        # Train
        for j in range(n_train_batches):
            bx = xtr_[j*batch_size:(j+1)*batch_size,:]
            by = ytr_[j*batch_size:(j+1)*batch_size,:]
            train_logger.accum(sess.run(train_to_run, {x:bx, y:by}))

        if (i+1) % 100 == 0:
            train_logger.print_(header='train', epoch=i+1,
                    time=time.time()-start, logfile=logfile)

            # Validation
            val_logger.clear()
            for j in range(n_val_batches):
                bx = xva[j*batch_size:(j+1)*batch_size,:]
                by = yva[j*batch_size:(j+1)*batch_size,:]
                val_logger.accum(sess.run(val_to_run, {x:bx, y:by}))
            val_logger.print_(header='val', epoch=i+1,
                    time=time.time()-start, logfile=logfile)

            print()
            logfile.write('\n')

    # Test
    logger = Accumulator('acc')
    for j in range(n_test_batches):
        bx = xte[j*batch_size:(j+1)*batch_size,:]
        by = yte[j*batch_size:(j+1)*batch_size,:]
        logger.accum(sess.run(tnet['acc'], {x:bx, y:by}))
    logger.print_(header='test')

    logfile.close()
    saver.save(sess, os.path.join(savedir, 'model'))

if __name__=='__main__':
    run()
