#!/bin/bash

sudo apt-get install libboost-all-dev
pip install scons

pushd $(mktemp -d)
git clone https://github.com/maestro-project/maestro.git
pushd ./maestro
scons
sudo mv maestro /usr/local/bin/maestro
popd
popd