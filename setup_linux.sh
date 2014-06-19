#!/bin/bash
# Setup CentOS 5
#
# Download CentOS-5.10-x86_64-bin-1of9.iso
# from http://isoredirect.centos.org/centos/5/isos/x86_64/
# 
# 

yum install wget gcc gcc-gfortran gcc-c++ \
unzip make zlib zlib-devel bzip2 \
openssh-server openssh-clients openssl-devel \
curl curl-devel expat expat-devel gettext \
-yq

wget https://github.com/git/git/archive/master.zip
mv master master.zip
unzip master.zip
cd git-master/
make prefix=/usr install
cd
rm -fr git-master/

wget http://repo.continuum.io/miniconda/Miniconda-3.0.0-Linux-x86_64.sh
sh Miniconda-3.0.0-Linux-x86_64.sh -b -p $HOME/anaconda

export PATH=$HOME/anaconda/bin:$PATH

git clone http://github.com/ezralanglois/arachnid/

conda install conda-build jinja2 setuptools binstar patchelf --yes
cd arachnid
cp .condarc ~/






