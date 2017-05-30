import tensorflow as tf
import numpy as np

def policy_network(observations,
                   init_states,
                   seq_len,
                   gru_unit_size,
                   num_layers,
                   num_actions):
    """ define policy neural network """
    with tf.variable_scope("rnn"):
        gru_cell = tf.contrib.rnn.GRUCell(gru_unit_size)
        gru_cells = tf.contrib.rnn.MultiRNNCell([gru_cell] * num_layers)
        output, final_state = tf.nn.dynamic_rnn(gru_cells, observations,
                initial_state=init_states, sequence_length=seq_len)

        output = tf.reshape(output, [-1, gru_unit_size])
        
    with tf.variable_scope("softmax"):
        w_softmax = tf.get_variable("w_softmax", shape=[gru_unit_size, num_actions],
            initializer=tf.contrib.layers.xavier_initializer())
        b_softmax = tf.get_variable("b_softmax", shape=[num_actions],
            initializer=tf.constant_initializer(0))

    logit = tf.matmul(output, w_softmax) + b_softmax
    return logit, final_state
