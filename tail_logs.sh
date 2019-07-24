#!/bin/sh
if [ -z "$1" ]; then
n=4;
else n=$1;
fi

while [ 1 ]; do
  for file in `ls hyperopt-mongo-worker-*.log`; do tail -n$n $file ; echo ; done
  echo
  sleep 1
done
