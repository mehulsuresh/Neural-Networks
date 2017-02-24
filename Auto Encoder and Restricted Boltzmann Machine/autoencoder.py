#AUTOENCODER IN TENSORFLOW
import tensorflow as tf
import numpy as np
import math


def autoencoder(dimensions=[784, 512, 256, 64]):
    #STORING PLACEHOLDER VALUES FOR INPUT
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    current_input = x
    #ENCODER
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(tf.random_uniform([n_input, n_output],-1.0 / math.sqrt(n_input),1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

   #INNER REPRESENTATION
    encoder.reverse()
    #DECODER
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    #OUTPUT OF THE NETWORK
    y = current_input

    # COST FUNCTION
    cost = tf.reduce_sum(tf.square(y - x))
    return {'x': x, 'y': y, 'cost': cost}


def autoencoder_mnist():

    import tensorflow as tf
    #IMPORT MNIST DATA
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    mean_img = np.mean(mnist.train.images, axis=0)
    myAutoencoder = autoencoder(dimensions=[784, 256, 64])

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(myAutoencoder['cost'])

    #CREATE SESSION
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    #RUN THE NETWORK
    #IT IS UNSUPERVISED AS WE ONLY PROVIDE ONE PARAMETER AND NO LABELS
    for epoch_i in range(20):
        for batch_i in range(mnist.train.num_examples // 50):
            batch_xs, _ = mnist.train.next_batch(50)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={myAutoencoder['x']: train})
        print(epoch_i, sess.run(myAutoencoder['cost'], feed_dict={myAutoencoder['x']: train}))


if __name__ == '__main__':
    autoencoder_mnist()