#!/bin/sh

python3 -m unidic download
pip3 freeze > /mnt/requirements.txt
bash wandb.sh

/usr/sbin/sshd -D
/bin/bash
