#!/bin/bash
set -e

echo "Git version:"
git --version

echo "installing boost tests from sources"
git clone https://github.com/boostorg/test boost-test
cd boost-test
./bootstrap.sh
sudo ./b2 --with-test install
#sudo ./b2 --with-test --prefix=$boost_installation_prefix install
cd ..
rm -rf boost-test
