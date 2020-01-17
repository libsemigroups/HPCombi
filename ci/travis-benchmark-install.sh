#!/bin/bash
set -e

echo "Git version:"
git --version

# echo "installing gtest from sources"
# git clone https://github.com/google/googletest.git
# cd googletest
# mkdir build
# cd build
# cmake ..
# make
# cd ../
# sudo cp -r googletest/include/gtest /usr/local/include
# sudo cp build/googlemock/gtest/lib*.a /usr/local/lib
# sudo cp build/googlemock/lib*.a /usr/local/lib
# cd ../
# rm -rf googletest

echo "installing benchmark from sources"
curl -L -O https://github.com/google/benchmark/archive/v1.5.0.tar.gz
tar xvf v1.5.0.tar.gz
cd benchmark-1.5.0
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON
make
sudo make install
cd ../../
rm -rf benchmark-1.5.0
