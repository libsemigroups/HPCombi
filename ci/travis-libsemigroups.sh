#!/bin/bash
set -e

echo "CMake version:"
cmake --version
echo "g++ version:"
$CXX --version
echo "gcc version:"
$CC --version

cd ..
git clone -b argcheck --depth=1 https://github.com/james-d-mitchell/libsemigroups.git
cd libsemigroups
mv ../HPCombi extern
echo "0.0.2" > extern/HPCombi/VERSION
./autogen.sh
./configure
make check -j2

