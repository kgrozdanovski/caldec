#!/bin/bash
# usage: targets.sh script.py kind...   -> prints non-IO ops with their target per kind
s=$1; shift
for k in "$@"; do
  ../rknn-venv/bin/python $s $k > $k.log 2>&1
  printf "%-16s " $k; awk '/Target InputShape/{t=1;next} t&&/^D RKNN: \S+ +[0-9]+ /{ if($5!="InputOperator"&&$5!="OutputOperator") printf "%s@%s ", $5, $7 } /Total Operator|^D RKNN: \S+ -{20}/{if(t>1)t=0; else if(t) t++}' $k.log; echo
done
