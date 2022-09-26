#!/bin/bash

campaign_results_path=$PWD/campaign_results
dname=$campaign_results_path/campaign_${1}
mkdir -p $dname
echo "Campaign "${1}" created using config.yaml"
cp config.yaml $dname

# for ii in $(seq 1 1 5); do
#     fname=$dname/readme.txt
#     echo "Campaign description" >> $fname
# done
