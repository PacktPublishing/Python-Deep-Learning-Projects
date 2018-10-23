#!/bin/bash

# Save this script to /home/yourUserName, chmod +x setupDeepLearning.sh, run using ./setupDeepLearning.sh

mkdir tensorflow
cd tensorflow

################################################################################
# Install utils.
################################################################################
echo -e "\e[36m***Installing utilities*** \e[0m"
sudo apt-get update
sudo apt-get install unzip git-all pkg-config zip g++ zlib1g-dev

################################################################################
# Install Java deps.
################################################################################
echo -e "\e[36m***Installing Java8. Press ENTER when prompted*** \e[0m"
echo -e "\e[36m***And accept licence*** \e[0m"
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer

################################################################################
# Install Bazel dep.
################################################################################
echo -e "\e[36m***Installing Bazel*** \e[0m"
wget https://github.com/bazelbuild/bazel/releases/download/0.11.1/bazel-0.11.1-without-jdk-installer-linux-x86_64.sh -O bazel-installer-linux-x86_64.sh
chmod +x bazel-installer-linux-x86_64.sh
sudo ./bazel-installer-linux-x86_64.sh
rm bazel-installer-linux-x86_64.sh
sudo chown $USER:$USER ~/.cache/bazel/

################################################################################
# Fetch Swig and Python deps.
################################################################################
echo -e "\e[36m***Installing python deps*** \e[0m"
sudo apt-get install swig
sudo apt-get install build-essential python-dev python-pip checkinstall
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev

################################################################################
# Fetch and install Python.
################################################################################
echo -e "\e[36m***Installing Python*** \e[0m"
wget https://www.python.org/ftp/python/2.7.14/Python-2.7.14.tgz
tar -xvf Python-2.7.14.tgz
cd Python-2.7.14
./configure
make
sudo make install
cd ../
rm Python-2.7.14.tgz

################################################################################
# Grab TensorFlow CPU version from central repo
################################################################################
echo -e "\e[36m***Cloning TensorFlow from GitHub*** \e[0m"
sudo pip install tensorflow

################################################################################
# Grab Keras from repo
################################################################################
echo -e "\e[36m***Installing Keras with Tensorflow backend*** \e[0m"
sudo pip install keras

################################################################################
# Installing other python dependencies
################################################################################
echo -e "\e[36m***Installing Python Deps*** \e[0m"
sudo apt-get install python-numpy
sudo pip install numpy --upgrade
sudo pip --no-cache-dir install Pillow pandas scipy sklearn
sudo pip install web.py gunicorn

echo -e "\e[36mReady to run TensorFlow! \e[0m"

################################################################################
# Checking the installations
################################################################################
echo -e "\e[36m***Listing modules*** \e[0m"

pip list
