from sigopt import Connection
import math

conn = Connection(client_token='TZCLEDVNUXIYPTVQRJTTKYYUZWKBPHCNOKIAHHQORJQHNKRG')
experiment = conn.experiments().create(
    name='Multi-Layer Perceptron',
    parameters=[
        {
            'name': 'num_hidden_1',
            'type': 'int',
            'bounds': {
                'min': 10,
                'max': 784,
            },
        {
            'name': 'num_hidden_2',
            'type': 'int',
            'bounds': {
                'min': 10,
                'max': 784,
            },
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
    observation_budget=70,
    metadata={
        'template': 'python_tf_mlp'
    }
)

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

def create_model(assignments, sess):
    weights = {
        'h1': tf.Variable(tf.random_normal([784, assignments['num_hidden_1']])),
        'h2': tf.Variable(tf.random_normal([assignments['num_hidden_1'], assignments['num_hidden_2']])),
        'out': tf.Variable(tf.random_normal([assignments['num_hidden_2'], 10])),
    }

    biases = {
        'h1': tf.Variable(tf.random_normal([assignments['num_hidden_1']])),
        'h2': tf.Variable(tf.random_normal([assignments['num_hidden_2']])),
        'out': tf.Variable(tf.random_normal([10])),
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=math.exp(assignments['log_learning_rate'])).minimize(cost)

    sess.run(tf.global_variables_initializer())

    batch_size = 100

    for i in range(assignments['epochs']):
        num_batches = int(mnist.train.num_examples / batch_size)

        for _ in range(num_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

    return out_layer


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
