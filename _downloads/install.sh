#!/bin/bash
####################################################################################
#
# This script installs Anaconda and Arachind in your current working directory.
# 
# $ sh install.sh
#
####################################################################################

##############################
# Download Miniconda installer
##############################

wget http://repo.continuum.io/miniconda/Miniconda-3.0.0-Linux-x86_64.sh

##############################
# Install in local directory
##############################

sh Miniconda-3.0.0-Linux-x86_64.sh -b -p $PWD/anaconda

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
echo "  - https://conda.binstar.org/public" >> $HOME/.condarc

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
conda install -c https://conda.binstar.org/ezralanglois arachnid

echo "Please ensure https://conda.binstar.org/public is in your $HOME/.condarc"
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

echo "If you have not already done so, please install SPIDER - http://spider.wadsworth.org/spider_doc/spider/docs/spi-register.html"



