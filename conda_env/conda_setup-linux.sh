# 
# Commands for Anaconda
# 

# list all conda environments
conda env list;

# remove env
conda env remove --name mip_v1;

# create env
conda env create -f mip_v1.yml

# activate package
echo "IMPORTANT - RUN: conda activate mip_v1";
