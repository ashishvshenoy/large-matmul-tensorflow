"""
A solution to finding trace of square of a large matrix using a single device.
We are able to circumvent OOM errors, by generating sub-matrices. TensorFlow
runtime, is able to schedule computation on small sub-matrices without
overflowing the available RAM.
"""

import tensorflow as tf
import os
import sys

tf.logging.set_verbosity(tf.logging.DEBUG)

N = 1000000 # dimension of the matrix
d = int(sys.argv[1]) # number of splits along one dimension. Thus, we will have 10000 blocks
d = 100
print "Value of d : "+str(d)
M = int(N / d)


def get_block_name(i, j):
    return "sub-matrix-"+str(i)+"-"+str(j)


def get_intermediate_trace_name(i, j):
    return "inter-"+str(i)+"-"+str(j)


# Create  a new graph in TensorFlow. A graph contains operators and their
# dependencies. Think of Graph in TensorFlow as a DAG. Graph is however, a more
# expressive structure. It can contain loops, conditional execution etc.
g = tf.Graph()

with g.as_default(): # make our graph the default graph
    tf.set_random_seed(1024)

    # in the following loop, we create operators that generate individual
    # sub-matrices as tensors. Operators and tensors are created using functions
    # like tf.random_uniform, tf.constant are automatically added to the default
    # graph.
    print "***Creating matrices***"
    matrices = {}
    count = 0
    for i in range(0, d):
        for j in range(0, d):
	    task_number = count%5
	    count = count+1
            with tf.device("/job:worker/task:%d" % task_number):
                matrix_name = get_block_name(i, j)
                matrices[matrix_name] = tf.random_uniform([M, M], name=matrix_name)

    intermediate_traces = {}
    count = 0
    print "***Calculating Traces***"
    for i in range(0, d):
        for j in range(0, d):
            task_number = count%5
            count = count+1
	    print "Iteration : "+str(count)
            with tf.device("/job:worker/task:%d" % task_number):
                A = matrices[get_block_name(i, j)]
                B = matrices[get_block_name(j, i)]
		trace_val = tf.trace(tf.matmul(A, B))
		#print "Trace Value : "+str(trace_val)
                intermediate_traces[get_intermediate_trace_name(i, j)] = trace_val

    # here, we add a "add_n" operator that takes output of the "trace" operators as
    # input and produces the "retval" output tensor.
    print "Aggregating the traces"
    with tf.device("/job:worker/task:0"):
        retval = tf.add_n(intermediate_traces.values())

    config = tf.ConfigProto(log_device_placement=True)
    with tf.Session("grpc://vm-4-1:2222", config=config) as sess:
        result = sess.run(retval)
	tf.train.SummaryWriter("%s/example_distributed" % (os.environ.get("TF_LOG_DIR")), sess.graph)
	sess.close()
	print "Trace of the big matrix is = ", result
