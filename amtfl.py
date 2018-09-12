from layers import *

def amtfl(x, y, training, nlist, alpha, gamma,
        mu=1e-4, lamb=1e-4, name='lenet', reuse=None):

    x = tf.reshape(x, [-1, 1, 28, 28])
    x = conv(x, 20, 5, name=name+'/conv1', reuse=reuse)
    x = relu(x)
    x = pool(x, name=name+'/pool1')
    x = conv(x, 50, 5, name=name+'/conv2', reuse=reuse)
    x = relu(x)
    x = pool(x, name=name+'/pool2')
    x = flatten(x)
    h = dense(x, 500, activation=relu, name=name+'/dense1', reuse=reuse)

    with tf.variable_scope(name + '/fc_last', reuse=reuse):
        S = tf.get_variable('SW', [500, 10],
                initializer=tf.random_normal_initializer(stddev=0.01))
        Sb = tf.get_variable('Sb', [10], initializer=tf.zeros_initializer())
        o = tf.matmul(h, S) + Sb

    count = get_count(nlist)
    cent = tf.reduce_mean(ovr_cross_entropy(o,y), 0)

    with tf.variable_scope(name + '/recon', reuse=reuse):
        AW = tf.get_variable('AW', [10, 500],
                initializer=tf.random_normal_initializer(stddev=0.01))
        Ab = tf.get_variable('Ab', [500], initializer=tf.zeros_initializer())
        hhat = relu(tf.matmul(o, AW) + Ab)

    aterm = 1 + alpha * tf.reduce_sum(tf.abs(AW), 1)
    proxy = tf.divide(0.001, count)
    cent_loss = tf.reduce_sum(aterm * (cent + proxy))

    recon_loss = gamma * tf.reduce_mean(tf.reduce_sum((h - hhat)**2, 1))

    net = {}
    all_vars = tf.get_collection('variables', scope=name)
    L = [v for v in all_vars if 'fc_last' not in v.name and 'recon' not in v.name]
    S = [v for v in all_vars if 'fc_last/SW' in v.name]
    net['weights'] = [v for v in all_vars]

    net['cent_loss'] = cent_loss
    net['recon_loss'] = recon_loss
    net['l1_decay'] = l1_decay(mu, var_list=S)
    net['l2_decay'] = l2_decay(lamb, var_list=L)
    net['acc'] = accuracy(o, y)
    return net
