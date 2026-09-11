#!/bin/bash
# Workaround for vastai/pytorch:cuda-12.8.1-auto: sshd refuses the (correct) key with
# "Authentication refused: bad ownership or modes for file $HOME/.ssh/authorized_keys".
# Re-assert ownership and modes, repeatedly for the first minutes in case the image's
# own entrypoint rewrites the file after us.
for i in $(seq 1 30); do
  chown -R root:root $HOME/.ssh 2>/dev/null
  chmod 700 $HOME/.ssh 2>/dev/null
  chmod 600 $HOME/.ssh/authorized_keys 2>/dev/null
  sleep 10
done &
echo "ssh-perm-fixer started" > /workspace/onstart.log 2>/dev/null || echo "ssh-perm-fixer started" > /tmp/onstart.log
