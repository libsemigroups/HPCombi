#!/bin/bash
set -e

echo "CMake version:"
cmake --version
echo "g++ version:"
$CXX --version
echo "gcc version:"
$CC --version

mkdir build
cd build
cmake -DBUILD_TESTING=1 -DCMAKE_BUILD_TYPE=Release ..
make
make test
cmake -DBUILD_TESTING=1 ..
make
make test
