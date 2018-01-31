#!/bin/bash
set -e

echo "Git version:"
git --version

git clone https://github.com/google/benchmark.git googlebenchmark
cd googlebenchmark
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make
sudo make install
rm -rf googlebenchmark
