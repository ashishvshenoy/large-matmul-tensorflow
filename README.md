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
