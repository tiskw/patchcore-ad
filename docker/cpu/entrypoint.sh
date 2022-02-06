#!/bin/bash -e

USERNAME="developer"

USRID=$(id -u)
GRPID=$(id -g)

# Create group.
if [ x"$GRPID" != x"0" ]; then
    groupadd -g ${GRPID} ${USERNAME}
fi

# Create user.
if [ x"$USRID" != x"0" ]; then
    useradd -d /home/${USERNAME} -m -s /bin/bash -u ${USRID} -g ${GRPID} ${USERNAME}
fi

# Restore permissions.
sudo chmod u-s /usr/sbin/useradd /usr/sbin/groupadd

export HOME="/home/${USERNAME}"

exec $@
