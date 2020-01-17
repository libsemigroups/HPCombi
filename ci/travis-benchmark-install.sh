#!/bin/bash
set -e

echo "Git version:"
git --version

# curl -L -O https://github.com/google/googletest/archive/release-1.10.0.tar.gz
# tar xvf release-1.10.0.tar.gz
# cd googletest-release-1.10.0
# mkdir build
# cd build
# cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON
# make
# sudo make install
# cd ../../
# rm -rf googletest-release-1.10.0

echo "installing benchmark from sources"
curl -L -O https://github.com/google/benchmark/archive/v1.5.0.tar.gz
tar xvf v1.5.0.tar.gz
cd benchmark-1.5.0
mkdir build
cd build
#cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DBENCHMARK_ENABLE_GTEST_TESTS=OFF -DBENCHMARK_ENABLE_TESTING=OFF
make
sudo make install
cd ../../
rm -rf benchmark-1.5.0
