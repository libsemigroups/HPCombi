#!/bin/bash
set -e

echo "Git version:"
git --version

echo "installing boost tests from sources"
wget --no-verbose --output-document=boost-trunk.tar.bz2 http://sourceforge.net/projects/boost/files/boost/1.60.0/boost_1_60_0.tar.bz2/download
export BOOST_ROOT="$TRAVIS_BUILD_DIR/../boost-trunk"
export CMAKE_MODULE_PATH="$BOOST_ROOT"
mkdir -p $BOOST_ROOT
tar jxf boost-trunk.tar.bz2 --strip-components=1 -C $BOOST_ROOT
(cd $BOOST_ROOT; ./bootstrap.sh --with-libraries=test)
(cd $BOOST_ROOT; ./b2 threading=multi --prefix=$BOOST_ROOT -d0 install)
