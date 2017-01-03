import tensorflow as tf
import os


tf.logging.set_verbosity(tf.logging.DEBUG)

N = 50
d = 5
M = int(N / d)


def get_block_name(i, j):
    return "sub-matrix-"+str(i)+"-"+str(j)


def get_intermediate_trace_name(i, j):
    return "inter-"+str(i)+"-"+str(j)

g = tf.Graph()

with g.as_default():
    tf.set_random_seed(1024)
    matrices = {}
    count = 0
    for i in range(0, d):
        for j in range(0, d):
            count = i*10+j
            with tf.device("/job:worker/task:%d" % count):
                matrix_name = get_block_name(i, j)
                matrices[matrix_name] = tf.random_uniform([M, M], name=matrix_name)

    intermediate_traces = {}
    count = 0
    for i in range(0, d):
        for j in range(0, d):
            count = i*10+j
            with tf.device("/job:worker/task:%d" % count):
                A = matrices[get_block_name(i, j)]
                B = matrices[get_block_name(j, i)]
                intermediate_traces[get_intermediate_trace_name(i, j)] = tf.trace(tf.matmul(A, B))

    with tf.device("/job:worker/task:0"):
        retval = tf.add_n(intermediate_traces.values())


    config = tf.ConfigProto(log_device_placement=True)
    with tf.Session("grpc://vm-4-1:2222",config=config) as sess:
        output = sess.run(retval) # executes all necessary operations to find value of retval tensor
        tf.train.SummaryWriter("%s/example_single" % (os.environ.get("TF_LOG_DIR")), sess.graph)
        sess.close()
        print "Trace of the big matrix is = ", output
