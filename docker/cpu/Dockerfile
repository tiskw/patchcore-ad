FROM ubuntu:focal

MAINTAINER Tetsuya Ishikawa <tiskw111@gmail.com>

# Set environment variables.
ENV DEBIAN_FRONTEND=noninteractive
ENV USERNAME=developer

# Copy and run the installer.
COPY installer.sh /installer.sh 
RUN sh installer.sh

# Copy a shell script for dynamic user creation.
COPY entrypoint.sh /entrypoint.sh

# Unlock permissions for the above "entrypoint.sh".
RUN chmod u+s /usr/sbin/useradd /usr/sbin/groupadd

# Set locales.
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

ENTRYPOINT ["sh", "/entrypoint.sh"]
CMD ["/bin/bash"]
