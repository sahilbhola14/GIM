#!/bin/bash
# Begin user input
system_name='H2_H'
# End user input

if [ -d $system_name'_system' ]; then
    echo "System $system_name already exists"

    read -p "Do you want to overwrite it? [y/n] " -n 1 -r

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        # echo -e ""
        echo "Overwriting system $system_name"
        rm -rf $system_name'_system'
    else
        echo "Exiting"
        exit 1
    fi

fi

echo "Creating system $system_name"
mkdir -p $system_name'_system'
cd $system_name'_system'
mkdir -p 'Kinetics_Data'
mkdir -p 'Cluster_Assignment/Energy_Based_Clustering'
for i in 1 2 4 8
do
    mkdir -p 'Cluster_Assignment/Energy_Based_Clustering/num_bins_'$i
done
