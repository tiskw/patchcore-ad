#!/bin/sh

# Update and upgrade installed packages.
apt-get update
apt-get upgrade -y
apt-get install -y apt-utils

# Install necessary packages.
apt-get install -y sudo locales

# Install Python3.
apt-get install -y python3 python3-dev python3-distutils python3-pip

# Install OpenCV.
apt-get install -y libopencv-dev

# Install Python packages.
pip3 install opencv-python==4.5.5.62 scikit-learn==1.0.2 rich==11.1.0 faiss-gpu==1.7.2 \
             torch==1.10.2+cu113 torchvision==0.11.3+cu113 thop==0.0.31-2005241907 \
             -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Set locale to UTF8.
locale-gen en_US.UTF-8  

# Clear package cache.
apt-get clean
rm -rf /var/lib/apt/lists/*

# Enable sudo without password.
mkdir -p /etc/sudoers.d
echo "${USERNAME} ALL=NOPASSWD: ALL" >> /etc/sudoers.d/${USERNAME}

# Unlock permissions.
chmod u+s /usr/sbin/useradd && chmod u+s /usr/sbin/groupadd
