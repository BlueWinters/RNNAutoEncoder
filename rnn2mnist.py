

import tensorflow as tf
import datafactory as df
import os as os
import numpy as np


# declare: a simple rnn for classification minst
# ref: https://www.jiqizhixin.com/articles/2017-09-29-7

data_set = 'mnist'
input_size = 784
n_class = 10

def get_config():
    data_path = 'D:/dataset/{}'.format(data_set)
    save_path = 'save/{}'.format(data_set)
    return data_path, save_path

def train():
    # time_steps = 28   # unrolled through 28 time steps
    n_hidden = 128    # hidden LSTM units
    n_input = 28      # rows of 28 pixels
    n_steps = 28      # time step

    n_epochs = 50000
    eph_step = 500
    learn_rate = 0.001
    batch_size = 100

    # placeholder for input and output
    x = tf.placeholder(tf.float32, [batch_size,n_steps,n_input])
    y = tf.placeholder(tf.float32, [batch_size,n_class])

    # weights and biases of appropriate shape to accomplish above task
    in_weights = tf.Variable(tf.random_normal([n_input,n_hidden]))
    in_bias = tf.Variable(tf.constant([0.1], shape=[n_hidden]))
    out_weights = tf.Variable(tf.random_normal([n_hidden,n_class]))
    out_bias = tf.Variable(tf.constant([0.1], shape=[n_class]))

    # cut input into pieces
    # for mnist, we make every rows of images as a input to step time of rnn
    input = tf.reshape(x, [batch_size*n_steps,n_input])
    # input of rnn cell
    rnn_in = tf.reshape(tf.matmul(input, in_weights)+in_bias,
                        shape=[batch_size,n_steps,n_hidden])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    rnn_out, final_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=init_state)
    outputs = tf.matmul(final_state[1], out_weights) + out_bias

    # loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels=y))
    solver = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    #
    correct_pred = tf.equal(tf.argmax(outputs,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    summary_op = tf.summary.merge_all()

    # load data
    data_path, save_path = get_config()
    tra, val = df.create_supervised_data(data_path, data_set, reshape=False, validation=True)

    # save directory
    if os.path.exists(save_path) == False:
        print('make dir ...')
        os.makedirs(save_path)

    #initialize variables
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(logdir=save_path, graph=sess.graph)
        log = open('{}/train.txt'.format(save_path), 'w')
        ave_loss = 0

        # training process
        for epochs in range(1,n_epochs+1):
            batch_x, batch_y = tra.next_batch(batch_size)
            batch_x = np.reshape(batch_x, [batch_size, n_steps, n_input])
            eph_loss, _ = sess.run([loss, solver], feed_dict={x:batch_x, y:batch_y})
            ave_loss += eph_loss/eph_step
            if epochs % eph_step == 0:
                acc = sess.run(accuracy, feed_dict={x:batch_x,y:batch_y})
                liner = 'epochs {}/{}, loss {}, accuracy {}'.format(epochs, n_epochs, ave_loss, acc)
                print(liner), log.writelines(liner+'\n')

        log.close()
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        writer = tf.train.Saver(vars)
        writer.save(sess, save_path='{}/model'.format(save_path))


if __name__ == '__main__':
    train()