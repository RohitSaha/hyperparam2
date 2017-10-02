from sigopt import Connection
import math

conn = Connection(client_token='TZCLEDVNUXIYPTVQRJTTKYYUZWKBPHCNOKIAHHQORJQHNKRG')
experiment = conn.experiments().create(
    name='Convolutional Neural Net',
    parameters=[
        {
            'name': 'conv1_kernel',
            'type': 'int',
            'bounds': {
                'min': 3,
                'max': 10,
            },
        },
        {
            'name': 'conv1_output',
            'type': 'int',
            'bounds': {
                'min': 10,
                'max': 49,
            },
        },
        {
            'name': 'conv1_act',
            'type': 'categorical',
            'categorical_values': [
                "relu",
                "sigmoid",
                "tanh",
            ]
        },
        {
            'name': 'conv2_kernel',
            'type': 'int',
            'bounds': {
                'min': 3,
                'max': 10,
            },
        },
        {
            'name': 'conv2_output',
            'type': 'int',
            'bounds': {
                'min': 10,
                'max': 49,
            },
        },
        {
            'name': 'conv2_act',
            'type': 'categorical',
            'categorical_values': [
                "relu",
                "sigmoid",
                "tanh",
            ]
        },
        {
            'name': 'fc1_hidden',
            'type': 'int',
            'bounds': {
                'min': 10,
                'max': 784,
            },
        },
        {
            'name': 'fc1_act',
            'type': 'categorical',
            'categorical_values': [
                "relu",
                "sigmoid",
                "tanh",
            ]
        },
        {
            'name': 'optimizer',
            'type': 'categorical',
            'categorical_values': [
                "adam",
                "rmsprop",
                "gradient_descent",
            ]
        },
        {
            'name': 'log_learning_rate',
            'type': 'double',
            'bounds': {
                'min': math.log(1e-8),
                'max': math.log(1),
            },
        },
        {
            'name': 'epochs',
            'type': 'int',
            'bounds': {
                'min': 5,
                'max': 20,
            },
        },
    ],
    observation_budget=10,
    metadata={
        'template': 'python_tf_cnn'
    }
)

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder("float", [None, 10])

activation_functions = {
    'relu': tf.nn.relu,
    'sigmoid': tf.sigmoid,
    'tanh': tf.tanh,
}

optimizers = {
    'gradient_descent': tf.train.GradientDescentOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
    'adam': tf.train.AdamOptimizer,
}

def create_model(assignments, sess):
    w_c1 = tf.Variable(tf.random_normal([assignments['conv1_kernel'], assignments['conv1_kernel'], 1, assignments['conv1_output']]))
    b_c1 = tf.Variable(tf.random_normal([assignments['conv1_output']]))
    conv1 = tf.nn.conv2d(x_img, w_c1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.add(conv1, b_c1)
    conv1 = tf.nn.max_pool(value=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = activation_functions[assignments['conv1_act']](conv1)

    w_c2 = tf.Variable(tf.random_normal([assignments['conv2_kernel'], assignments['conv2_kernel'], assignments['conv1_output'], assignments['conv2_output']]))
    b_c2 = tf.Variable(tf.random_normal([assignments['conv2_output']]))
    conv2 = tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.add(conv2, b_c2)
    conv2 = activation_functions[assignments['conv2_act']](conv2)

    conv2_flat = tf.contrib.layers.flatten(conv2)

    w_fc1 = tf.Variable(tf.random_normal([conv2_flat.get_shape()[1].value, assignments['fc1_hidden']]))
    b_fc1 = tf.Variable(tf.random_normal([assignments['fc1_hidden']]))
    fc1 = tf.add(tf.matmul(conv2_flat, w_fc1), b_fc1)
    fc1 = activation_functions[assignments['fc1_act']](fc1)

    w_out = tf.Variable(tf.random_normal([assignments['fc1_hidden'], 10]))
    b_out = tf.Variable(tf.random_normal([10]))
    out = tf.add(tf.matmul(fc1, w_out), b_out)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
    optimizer = optimizers[assignments['optimizer']](learning_rate=math.exp(assignments['log_learning_rate'])).minimize(cost)

    sess.run(tf.global_variables_initializer())

    batch_size = 100

    for i in range(assignments['epochs']):
        num_batches = int(mnist.train.num_examples / batch_size)

        for _ in range(num_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

    return out

def evaluate_model(assignments):
    with tf.Session() as sess:
        model = create_model(assignments, sess)

        correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
    return float(acc)

for _ in range(experiment.observation_budget):
    suggestion = conn.experiments(experiment.id).suggestions().create()
    assignments = suggestion.assignments
    value = evaluate_model(assignments)

    conn.experiments(experiment.id).observations().create(
        suggestion=suggestion.id,
        value=value
    )

assignments = conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments

print(assignments)

# This is a SigOpt-tuned model
with tf.Session() as sess:
    classifier.create_model(assignments, sess)
