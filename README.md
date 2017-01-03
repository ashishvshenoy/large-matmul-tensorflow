# Large Matrix Multiplication with TensorFlow
This tensorflow python program multiplies a randomly generated 100000x100000 matrix and calculates the trace of the result.
The distributed execution of this mathematical operation is designed as follows :
* Generate a random 10000x10000 matrix on the devices in round-robin fashion.
* Multiply the matrices in the same round-robin fashion on all the devices.
* Calculate the intermediate trace.
* The above logic is repeated 10000 times.
* Aggregate all the intermediate traces on a centralized device.

## Running Method
`./bigmatrixmultiplication.sh or - python bigmatrixmultiplication.py`

## exampleMatmulSingle.py, exampleMatmulFailure.py and exampleMatmulDistributed.py
These are different variations of the solutions to the same problem of multiplying a very large matrix. The Single version is an implementation on a single node, whereas Distributed is an implementation on a 5 node cluster.

## Details about the environment used
A 5 node cluster, each node with 20GB RAM and 4 cores was used to run this application.
