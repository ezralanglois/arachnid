#!/bin/bash
####################################################################################
#
# This script installs Anaconda and Arachnid in your current working directory.
# 
# $ sh install.sh
#
# The default install is the latest stable version of Arachnid. 
#
# If you wish the latest accelerated version (a premium package 
# free for Academic use) then use the following command:
#
# $ sh install.sh mkl
#
# If you wish the latest daily build, then using this command:
#
# $ sh install.sh dev
#
# If you wish the latest accelerated daily build, then use this command:
# 
# $ sh install.sh dev-mkl
#
####################################################################################

##############################
# Handle command line options
##############################

suffix=""
if [ "$1" != "" ] ; then

if [ "$1" == "-h" ] || [ "$1" == "--help" ] || [ "$1" == "-help" ] ; then
head -23 $0
exit 0
fi

if [ "$1" != "mkl" ] && [ "$1" != "dev" ] && [ "$1" != "dev-mkl" ] ; then

echo "Error: unrecognized option $1"
echo "Only mkl, dev, dev-mkl and empty string supported"
echo "sh install -h for more information"
exit 1

fi

suffix="-${1}"

fi

##############################
# Test if Anaconda exists
##############################

which conda
if [ $? -eq 0 ] ; then
echo "You already have anaconda installed"
echo "You can use the following command to install Arachnid"
echo "conda install -c https://conda.binstar.org/ezralanglois arachnid${suffix} --yes"
exit 1
fi

##############################
# Download Miniconda installer
##############################

wget http://repo.continuum.io/miniconda/Miniconda-3.0.0-Linux-x86_64.sh

if [ $? -ne 0 ] ; then
echo "Failed to download Anaconda"
exit 1
fi

##############################
# Install in local directory
##############################

sh Miniconda-3.0.0-Linux-x86_64.sh -b -p $PWD/anaconda

if [ $? -ne 0 ] ; then
echo "Failed to install Anaconda"
exit 1
fi

##############################
# Remove installer
##############################
rm -f Miniconda-3.0.0-Linux-x86_64.sh

##############################
# Place Anaconda on the PATH for the duration of the script
##############################
export PATH=$PWD/anaconda/bin:$PATH

##############################
# Test if $HOME/.condarc exists
##############################

if [ ! -e $HOME/.condarc ] ; then

##############################
# If not, create one
##############################

echo "No $HOME/.condarc found, creating one ... "

echo "channels:" > $HOME/.condarc
echo "  - http://repo.continuum.io/pkgs/pro" >> $HOME/.condarc
echo "  - http://repo.continuum.io/pkgs/free" >> $HOME/.condarc
echo "  - http://repo.continuum.io/pkgs/gpl" >> $HOME/.condarc
echo "  - https://conda.binstar.org/ezralanglois" >> $HOME/.condarc

##############################
# Install Arachnid
##############################
conda install arachnid

echo "You will now be able to update arachnid with: conda update arachnid"

else

##############################
# If so, skipping creating one
##############################

echo "Found $HOME/.condarc found"

##############################
# Install Arachnid
##############################

# Install arachnid
conda install -c https://conda.binstar.org/ezralanglois arachnid${suffix} --yes

echo "Please ensure https://conda.binstar.org/ezralanglois is in your $HOME/.condarc"
echo "Then you will be able to update with: conda update arachnid"

fi

###################################
# Test if shell is other than bash
###################################
shell=`basename $SHELL`
if [ "$shell" != "bash" ]; then
	echo "Assuming you have CSH or TCSH"
	echo "Appending $PWD/anaconda/bin to PATH"
	echo "setenv PATH \"$PWD/anaconda/bin:\$PATH\"" >> $HOME/.${shell}rc
fi

if [ $suffix == "" ] || [ $suffix == "mkl" ] ; then
echo "If you have not already done so, please install SPIDER - http://spider.wadsworth.org/spider_doc/spider/docs/spi-register.html"
fi



