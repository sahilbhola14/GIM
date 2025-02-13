#!/bin/bash

campaign_results_path=$PWD/campaign_results
dname=$campaign_results_path/campaign_${1}

if [[ -d $dname ]]; then

    read -p "Campaign already exists! Recreate it? [y/yes/Y] : " flag

    if [[ $flag == "y" || $flag == "yes" || $flag == "Y" ]]; then

        rm -rf $dname
        mkdir -p $dname
        fname=$dname/readme.txt
        fig_file_name=$dname/Figures
        mkdir -p $fig_file_name
        echo "Campaign description" >> $fname
        cp config.yaml $dname
        echo "Campaign "${1}" (Re)created using config.yaml"

        # Generating the data from using config file
        python generate_training_data.py ${1}

    else

        echo "Campaign "${1}" was not removed"

    fi

else

    mkdir -p $dname
    fname=$dname/readme.txt
    fig_file_name=$dname/Figures
    mkdir -p $fig_file_name
    echo "Campaign description" >> $fname
    cp config.yaml $dname
    echo "Campaign "${1}" created using config.yaml"

    # Generating the data from using config file
    python generate_training_data.py ${1}


fi
