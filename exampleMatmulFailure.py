"""
An example showing why we cannot directly compute the trace of product of large
matrices. This script will throw Out-of-memory (OOM) errors.
"""

import tensorflow as tf
import os

# the next two lines should be self-explanatory
tf.logging.set_verbosity(tf.logging.DEBUG)
tf.set_random_seed(1024)

N = 100000

# Create  a new graph in TensorFlow. A graph contains operators and their
# dependencies. Think of Graph in TensorFlow as a DAG. Graph is however, a more
# expressive structure. It can contain loops, conditional execution etc.
g = tf.Graph()

with g.as_default(): # make our graph as the defaul graph

    # We create an operator that generates the big NxN matrix as a single
    # tensor. Operators and tensors are created using functions like
    # tf.random_normal and tf.constant are automatically added to the default
    # graph.
    A = tf.random_uniform([N, N], name="big_matrix")

    # create a single "matmul" and "trace" operator
    bigtrace = tf.trace(tf.matmul(A, A))


# Here, we create session. A session is required to run a computation
# represented as a graph.
sess = tf.Session(graph=g)

# executes all necessary operations to find value of bigtrace tensor
# Note that script will throw OOM error only when run is executed. Try
# commenting it out.
output = sess.run(bigtrace)

# Summary writer is used to write the summary of execution including graph
# structure into a log directory. By pointing "tensorboard" to this directory,
# we will be able to graphically view the graph. The constructor for summary
# writer takes as input the log director and the graph used in the session.
tf.train.SummaryWriter("%s/example_single" % (os.environ.get("TF_LOG_DIR")), sess.graph)

sess.close()

print "Trace of the big matrix is = ", output
