#!/bin/bash

NUM=50
VAL=10
for i in `seq 1 $NUM`; do
  echo $VAL
  $1 $VAL $VAL $VAL $VAL $VAL
  VAL=$(( $VAL + 10 ))
done
