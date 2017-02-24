#RESTRICTIVE BOLTZMANN MACHINE IN PYTHON
import numpy as np
from numpy.random import choice, uniform, randn
import input_data

#SIGMOID ACTIVATION FUNCTION
def sigmoid(x):
    return 1. / (1. + np.exp(-1. * x))

# FIND VALIDATION ERROR
def validation_error(valid_data, weights, hid_biases, vis_biases):
    # FORWARD/UP
    hid_input = np.dot(valid_data, weights) + hid_biases
    hid_states = sigmoid(hid_input) > np.random.uniform(size=hid_input.shape)
    # BACKWARD/DOWN
    vis_probs = sigmoid(np.dot(hid_states, weights.T) + vis_biases)
    valid_err = np.sum((valid_data - vis_probs) ** 2) / valid_data.shape[0]
    return valid_err, vis_probs

# NUMBER OF VISIBLE LAYERS
vis_layers = 784
# NUMBER OF HIDDEN LAYERS
hid_layers = 250
#INITIAL PARAMETERS
n_epochs = 6
batch_size = 50
learning_rate = 0.1
momentum = 0.5
decay = 0.0002

#INSERTING PLACEHOLDER VALUES INTO INITIAL VARIABLES
weights = 0.001 * randn(vis_layers, hid_layers)
hid_biases = np.zeros((1, hid_layers))
vis_biases = np.zeros((1, vis_layers))
#FOR MOMENTUM
weights_temp = np.zeros((vis_layers, hid_layers))
hid_biases_temp = np.zeros((1, hid_layers))
vis_biases_temp = np.zeros((1, vis_layers))


#LOADING MNIST DATA SET AND IT IS FIRST CONVERTED TO B/W SO WE CAN USE WITH BINARY BOLTZMANN MACHINE
data_sets = input_data.read_data_sets('MNIST_data', one_hot=True, greyscale=False)

#NUMBER OF MINIBATCHES
n_minibatches = data_sets.train.num_minibatches(batch_size)


for i_epoch in range(n_epochs):#Contrastive Divergence (CD-k)
    train_err = 0.0

    for i_minibatch in range(n_minibatches):
        batch = data_sets.train.next_batch(batch_size, no_labels=True)
        
        #FORWARD PROPAGATION FIND HIDDEN LAYERS
        fwd_hid_probs = sigmoid(np.dot(batch, weights) + hid_biases)
        #BINARY CONVERSION
        fwd_hid_states = fwd_hid_probs > uniform(size=fwd_hid_probs.shape)
        fwd_prods = np.dot(batch.T, fwd_hid_probs)
        fwd_hid_act = np.sum(fwd_hid_probs, axis=0)
        fwd_vis_act = np.sum(batch, axis=0)


        # BACKWARD PROPAGATION FIND THE INPUT LAYERS
        back_data = sigmoid(np.dot(fwd_hid_states, weights.T) + vis_biases)
        back_hid_probs = sigmoid(np.dot(back_data, weights) + hid_biases)
        back_prods = np.dot(back_data.T, back_hid_probs)
        back_hid_act = np.sum(back_hid_probs, axis=0)
        back_vis_act = np.sum(back_data, axis=0)

        #FIND CHANGES IN WEIGHTS AND BIASES
        weights_delta = (fwd_prods - back_prods)/batch_size
        vis_biases_delta = (fwd_vis_act - back_vis_act)/batch_size
        hid_biases_delta = (fwd_hid_act - back_hid_act)/batch_size

        # MOMENTUM + DECAY
        weights += learning_rate * weights_delta - decay *weights + momentum * weights_temp

        # MOMENTUM
        vis_biases += learning_rate * vis_biases_delta + momentum * vis_biases_temp
        hid_biases += learning_rate * hid_biases_delta  + momentum * hid_biases_temp

        # SAVE COPIES FOR MOMENTUM
        weights_temp = weights_delta
        vis_biases_temp = vis_biases_delta
        hid_biases_temp = hid_biases_delta

        #TRAINING ERROR
        train_err += np.sum((batch - back_data) ** 2)


    # FIND TRAINING ERROR
    train_err /= batch_size * n_minibatches
    # FIND VALIDATION ERROR
    valid_err, valid_vis_probs = validation_error(data_sets.validation.images, weights, hid_biases, vis_biases)
    print ("Epoch %i \nTraining Error: %.3f\nValidation Error: %.3f" % (i_epoch, train_err, valid_err))
    
    