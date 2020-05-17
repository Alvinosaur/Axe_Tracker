import csv
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import random
import sys
import file_helpers

NUM_PIX = 700 * 600
NUM_OUTPUTS = 4
OUTPUTS = [0, 1, 3, 5]

ignore_file = ".DS_Store"

def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res

def soft_max(x):
    res = np.exp(x)
    normalizer = np.sum(res)  # following law of total probabilities
    return res / np.sum(res)

def prepend_bias(mat, val, is_mat):
    return np.insert(mat, 0, val, axis=is_mat)


def feed_forward(alpha, beta, x):
    # sigmoid activation
    z = sigmoid(np.dot(alpha, x))
    # add 1 again for bias term
    z = prepend_bias(z, 1, is_mat=False)

    # final output uses  softmax
    y_hat = soft_max(np.dot(beta, z))

    # generate one-hot vector for class prediction
    return z, y_hat

def backpropagation(x, z, beta, y_hat, y):
    # dl_dbeta, z includes bias here
    dl_db = y_hat - y
    dl_dbeta = np.dot(dl_db, z.T)

    # z shouldn't include bias here
    z_no_bias = z[1:]
    dl_dz = np.dot(beta.T, dl_db)
    dl_da = np.multiply(dl_dz, np.multiply(z_no_bias, (1 - z_no_bias)))

    dl_dalpha = np.dot(dl_da, x.T)
    return dl_dalpha, dl_dbeta


def loss(y_hat, y):
    loss = np.dot(y.T, np.log(y_hat))
    return -1 * loss[0][0]


def init_weights(is_rand, shape):
    # clamp [0, 1] => [-0.5, 0.5] => [-0.1, 0.1]
    if is_rand: weights = (np.random.rand(shape[0], shape[1]) - 0.5) / 5.0
    else: weights = np.zeros(shape)
    return weights

def get_features_output(img_file, K=NUM_OUTPUTS):
    y_val = file_helpers.parse_true_label(img_file)
    try:
        y = np.zeros((K, 1))
        y[OUTPUTS.index(y_val)][0] = 1
    except Exception as e:
        print("Failed to load true value b/c %s for img, %s" % (e, img_file))

    # create feature vector
    x = file_helpers.load_image(img_file)
    if x is None:
        print("Feature X was NONE, so image %s does not exist" % img_file)
        return None, None
    x = prepend_bias(x, 1, is_mat=False)
    return x, y


def y_hat_one_hot_encoding(y_hat):
    max_i = np.argmax(y_hat)
    one_hot_y_hat = np.zeros(y_hat.shape)
    one_hot_y_hat[max_i][0] = 1
    return one_hot_y_hat


def train(num_epoch, D, K, M, is_rand, learning_rate, train_data, 
        validation_data):
    """Uses Stochastic Gradient Step to update weights every training example.
    
    Arguments:
        num_epoch {[type]} -- [description]
        D {int} -- num hidden units in  hidden layer
        K {int} -- num output classes
        M {int} -- num features per training example
        is_rand {bool} -- init with random weights U[-0.1, 0.1] or 0
        learning_rate {float} -- learning rate
        data {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    alpha = init_weights(is_rand, (D, M))
    # always init bias to 0
    alpha = prepend_bias(alpha, 0, is_mat=True)
    beta = init_weights(is_rand, (K, D))  # +1 for bias: x0 = 1
    beta = prepend_bias(beta, 0, is_mat=True)
    train_cost = []
    validation_cost = []
    epoch_count = 0
    for epoch in range(num_epoch):
        for example in train_data:
            if example == ignore_file: continue
            # create one-hot vector for ground truth
            x, y = get_features_output(example, K)
            if x is None or y is None: continue

            # Generate prediction and hidden layer output for backprop
            z, y_hat = feed_forward(alpha, beta, x)

            # Perform backprop to get gradients
            g_alpha, g_beta = backpropagation(x, z, beta[:, 1:], y_hat, y)
            alpha -= learning_rate * g_alpha
            beta -= learning_rate * g_beta
        
        # show error rates for first two epochs
        # train_error, _ = cross_entropy(train_data, alpha, beta, M)
        # test_error, _ = cross_entropy(validation_data, alpha, beta, M)
        # print(train_error)
        # train_cost.append(train_error)
        # validation_cost.append(test_error)
    return alpha, beta, train_cost, validation_cost


def cross_entropy(data, alpha, beta, M, K=NUM_OUTPUTS):
    mean_entropy = 0
    error_count =  0
    for example in data:
        if example == ignore_file: continue
        # create one-hot vector for ground truth
        x, y = get_features_output(example, K)
        z, y_hat = feed_forward(alpha, beta, x)
        mean_entropy += loss(y_hat, y)
        true_label = int(example[-5])
        true_index = OUTPUTS.index(true_label)
        # count how many wrong predictions
        error_count += (true_index != int(np.argmax(y_hat)))
    return mean_entropy / float(len(data)), error_count / float(len(data))


def gen_metrics_output(train_data, test_data, alpha, 
        beta, train_error, test_error, M, K):
    _, final_train_error = cross_entropy(train_data, alpha, beta, M, K)
    _, final_test_error = cross_entropy(test_data, alpha, beta,  M, K)
    output = ""
    for i in range(len(train_error)):
        output += "epoch=%d crossentropy(train): %f\n" % (i+1, train_error[i])
        output += "epoch=%d crossentropy(test): %f\n" % (i+1, test_error[i])

    output += "error(train): %f\n" % final_train_error
    output += "error(test): %f\n" % final_test_error
    return output


def plot_results(train_error, test_error, num_epoch, hidden_units, 
        learning_rate):
    x_axis = np.arange(len(train_error)+1)[1:]
    data = [(x_axis, train_error), (x_axis, test_error)]
    colors = ("red", "green")
    groups = ("train error", "test error")

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, c=color, s=30, label=group)

    plt.title('Train and Test error at %f Learning Rate v.s num epochs' % 
        learning_rate)
    plt.legend(loc=1)
    plt.savefig('epochs(%d)_hidden_units(%d)_learn_rate(%f).png' % (
        num_epoch, hidden_units, learning_rate), bbox_inches='tight')


def main():
    # Parse commandline arguments
    train_input_filepath = sys.argv[1]
    # test_input_filepath = sys.argv[2]
    # test_output_filepath = sys.argv[4]
    weights_output = sys.argv[2]
    num_epoch = int(sys.argv[3])
    hidden_units = int(sys.argv[4])
    init_flag = int(sys.argv[5])
    is_rand = (init_flag == 1)
    learning_rate =  float(sys.argv[6])
    metrics_output_file = sys.argv[7]

    # Extract data
    
    all_imgs = file_helpers.get_img_names(train_input_filepath)
    random.shuffle(all_imgs)
    partition = int(0.8 *  len(all_imgs))
    train_imgs = all_imgs[:partition]
    test_imgs = all_imgs[partition:]
    x, _ = get_features_output(train_imgs[0])
    num_features = x.shape[0] - 1

    alpha, beta, train_error, test_error = train(num_epoch, hidden_units, 
        NUM_OUTPUTS, num_features, is_rand, learning_rate, train_imgs, test_imgs)
    train_predictions = []

    with open(weights_output, "w+") as f:
        weights_alpha = [str(val) for val in alpha.reshape(-1)]
        weights_beta = [str(val) for val in beta.reshape(-1)]
        weights_str = ",".join(list(weights_alpha))
        weights_str += "\n"
        weights_str += ",".join(list(weights_beta))
        weights_str += "\n"
        f.write(weights_str)

    # for example in train_imgs:
    #     x, y = get_features_output(example, num_features, NUM_OUTPUTS)
    #     z, y_hat = feed_forward(alpha, beta, x)
    #     prediction = str(np.argmax(y_hat))
    #     train_predictions.append("Img: %s, Label: %s" % (example, prediction))

    # with open(train_output_filepath, "w+") as f:
    #     f.write("\n".join(train_predictions))
    
    # test_predictions = []
    # for example in test_data:
    #     x, _ = get_features_output(example, num_features, NUM_OUTPUTS)
    #     z, y_hat = feed_forward(alpha, beta, x)
    #     prediction = str(np.argmax(y_hat))
    #     test_predictions.append(prediction)
    # with open(test_output_filepath, "w+") as f:
    #     f.write("\n".join(test_predictions))
    
    metrics_output = gen_metrics_output(train_imgs, test_imgs, alpha, 
        beta, train_error, test_error, num_features, NUM_OUTPUTS)
    with open(metrics_output_file, "w+") as f:
        f.write(metrics_output)

    # plot_results(train_error, test_error, num_epoch, hidden_units, 
    #     learning_rate)


if __name__ == "__main__":

    main()

# Example Run:
"""
python neuralnet.py ~/Documents/axe_images/positives 50 20 1 0.001
"""