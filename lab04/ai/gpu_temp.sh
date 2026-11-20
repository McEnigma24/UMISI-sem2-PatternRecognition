#!/bin/bash

clear

nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits -l 1 | awk -F',' '{
  id=$1+0; t=$2+0;
  if(!(id in min) || t<min[id]) min[id]=t;
  if(!(id in max) || t>max[id]) max[id]=t;
  cur[id]=t;
  if(id==1) {
    printf "\rGPU0: %2d°C (Min: %2d | Max: %2d)  ||  GPU1: %2d°C (Min: %2d | Max: %2d)", cur[0], min[0], max[0], cur[1], min[1], max[1];
    fflush();
  }
}'


