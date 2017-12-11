# Building HPCombi

## Build Prerequisites:

- CMake 2.8 or later

- A recent c++ compiler. I have tested the code on
  * g++ 5.3.1, 6.2.1 and 7.1.1.
  * clang 5.0.0
  * g++ 4.8 and 4.9 are known to be broken (I can fix it if needed at the price
  of some uglification of the code).

- [optional] : Google sparsehash/dense_hash_map, sparsehash/dense_hash_set.
  if not the less efficient standard containers will be used.

- BOOST.test (shared library version) : needed for testing.

- Your machine must support AVX instructions.

- Doxygen for generating the API documentation (in progress).

## Building

Using Make:

    mkdir build
    cd build
    cmake ..
    make

If you want to build the tests:

    mkdir build
    cd build
    cmake -DBUILD_TESTING=1 ..
    make
    make test

By default, cmake compile in debug mode (no optimisation, assert are on). To
compile in release mode:

    cmake -DCMAKE_BUILD_TYPE=Release ..
