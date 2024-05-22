# flow-counting

A reference implementation of the flow-counting algorithm for a hybrid pixel detector.

![build](https://github.com/m31k0r/flow-counting/actions/workflows/rust.yml/badge.svg)

## Implementation Detail

For the general idea of the algorithm see our publication in [Ultramicroscopy](https://doi.org/10.1016/j.ultramic.2023.113864)

To minimize costly moves, we merged the output and buffer to one vector, and only compare with the *B* last elements in the output.

If you want to deploy this algorithm in a real-time toolchain, you only have to change the input vector *hits* and the output vector *extracted_cluster* to a single-consumer-single-producer buffer. (Mind, that it should be optimized to allow a fast enough memory throughput).

## Usage

Build it like any other Rust program with cargo and provide a suitable hdf5 file. The software looks for the keys:

key | description
----|------------
x   | x pixel of the hit
y   | y pixel of the hit
tot | time over threshold
toa | time of arrival
