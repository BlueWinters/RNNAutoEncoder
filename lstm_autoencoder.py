"""
title: LSTM-AutoEncoder
"""
import tensorflow as tf
import datafactory as df
import os as os
import numpy as np


data_set = 'mnist'
input_size = 784

def get_config():
    data_path = 'D:/dataset/{}'.format(data_set)
    save_path = 'save/{}'.format(data_set)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    return data_path, save_path

def train():

    n_hidden = 1000    # hidden LSTM units
    n_input = 28      # rows of 28 pixels
    n_steps = 28      # sum time step

    n_epochs = 50000
    eph_step = 500
    learn_rate = 0.001
    batch_size = 100


    # placeholder for input and output
    x = tf.placeholder(tf.float32, [batch_size,n_steps,n_input])

    in_w = tf.Variable(tf.random_normal([n_input,n_hidden]), name='in_w')
    in_b = tf.Variable(tf.constant([0.1], shape=[n_hidden]), name='in_b')
    out_w = tf.Variable(tf.random_normal([n_hidden,n_input]), name='out_w')
    out_b = tf.Variable(tf.constant([0.1], shape=[n_input]), name='out_b')

    # input*W_in+b_in -> rnn_in
    input = tf.reshape(x, [batch_size*n_steps,n_input])
    input = tf.sigmoid(tf.matmul(input, in_w)+in_b)
    rnn_in = tf.reshape(input, shape=[batch_size,n_steps,n_hidden])

    # rnn_in -> lstm_cell -> rnn_out
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    rnn_out, _ = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=init_state)

    # rnn_out*W_out+b_out -> output
    rnn_out = tf.reshape(rnn_out, shape=[batch_size*n_steps,n_hidden])
    output = tf.sigmoid(tf.matmul(rnn_out, out_w) + out_b)
    output = tf.reshape(output, shape=[batch_size,n_steps,n_input])
    y = output[:,-n_input:,:]

    # loss
    loss = tf.reduce_mean(tf.square(x-y))
    solver = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    # load data and config
    data_path, save_path = get_config()
    data = df.create_supervised_data(data_path, data_set, reshape=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        log = open('{}/train.txt'.format(save_path), 'w')
        ave_loss = 0

        # training process
        for epochs in range(1,n_epochs+1):
            batch_x, _ = data.next_batch(batch_size)
            batch_x = np.reshape(batch_x, [batch_size, n_steps, n_input])
            eph_loss, _ = sess.run([loss, solver], feed_dict={x:batch_x})
            ave_loss += eph_loss/eph_step
            if epochs % eph_step == 0:
                liner = 'epochs {}/{}, loss {}'.format(epochs, n_epochs, ave_loss)
                print(liner), log.writelines(liner+'\n')
                ave_loss = 0

        log.close()
        # save models
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        writer = tf.train.Saver(vars)
        writer.save(sess, save_path='{}/model'.format(save_path))



if __name__ == '__main__':
    train()