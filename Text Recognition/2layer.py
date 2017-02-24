#CODE TO IMPORT THE MNIST DATABASE
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#START TENSOR FLOW SESSION
import tensorflow as tf
sess = tf.InteractiveSession()

#INITIALIZE WEIGHTS
def weight_init(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#INITIALIZE BIASES
def bias_init(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#CONVOLUTION 
#STRIDE : 1(horizontal & vertical)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#MAX POOLING
#2 X 2
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#FIRST CONVOLUTION LAYER
#KERNEL SIZE 5 X 5
#NUMBER OF FEATURES 32
#INPUT IS 1 IMAGE
weights_conv_layer_1 = weight_init([5, 5, 1, 32])
biases_conv_layer_1 = bias_init([32])

x_image = tf.reshape(x, [-1,28,28,1])

conv_layer_1 = tf.nn.relu(conv2d(x_image, weights_conv_layer_1) + biases_conv_layer_1)
pooling_layer_1 = max_pool_2x2(conv_layer_1)

#Visualization of First Convolution Layer
channels = 32
img_size = 28

def visualize_filter(index):
  
  V = tf.slice(conv_layer_1, (index, 0, 0, 0), (1, -1, -1, -1))
  V = tf.reshape(V, (img_size, img_size, channels))

  V = tf.transpose(V, (2, 0, 1))
  V = tf.reshape(V, (-1, img_size, img_size, 1))

  tf.image_summary("first_conv_filter_" + str(index), V)

[visualize_filter(i) for i in range(32)]

#SECOND CONVOLUTION LAYER
#KERNEL SIZE 5 X 5
#NUMBER OF FEATURES 64
#INPUT IS 32 IMAGES AFTER CONVOLUTION AND POOLING
weights_conv_layer_2 = weight_init([5, 5, 32, 64])
biases_conv_layer_2 = bias_init([64])

conv_layer_2 = tf.nn.relu(conv2d(pooling_layer_1, weights_conv_layer_2) + biases_conv_layer_2)

#MAX POOLING 
pooling_layer_2 = max_pool_2x2(conv_layer_2)

#FULLY CONNECTED LAYER 1
#100 NEURONS
weights_fully_connected_layer_1 = weight_init([7 * 7 * 64, 1024])
biases_fully_connected_layer_1 = bias_init([1024])

pooling_layer_2_flat = tf.reshape(pooling_layer_2, [-1, 7*7*64])
fully_connected_layer_1 = tf.nn.relu(tf.matmul(pooling_layer_2_flat, weights_fully_connected_layer_1) + biases_fully_connected_layer_1)

#DROPOUT
keep_prob = tf.placeholder(tf.float32)
fully_connected_layer_1_drop = tf.nn.dropout(fully_connected_layer_1, keep_prob)

#FULLY CONNECTED LAYER 2
#10 NEURONS
#SOFTMAX LAYER
#10 OUTPUTS
weights_fully_connected_layer_2 = weight_init([1024, 10])
biases_fully_connected_layer_2 = bias_init([10])

y_conv=tf.nn.softmax(tf.matmul(fully_connected_layer_1_drop, weights_fully_connected_layer_2) + biases_fully_connected_layer_2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/2layer_logs", sess.graph_def)
sess.run(tf.initialize_all_variables())

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    feed_dict = {x:batch[0], y_: batch[1], keep_prob: 1.0}
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    summary_str = sess.run(merged, feed_dict)
    writer.add_summary(summary_str, i)
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))