# 
# Commands for Anaconda
# 

# list all conda environments
conda env list;

# remove env
conda env remove --name mip_v1;

# create env
conda create -n mip_v1 --channel anaconda python=3.8;

# install packages 
conda install --channel anaconda -n mip_v1 pandas scrapy numpy \
	psycopg2 spacy nltk matplotlib scikit-learn scipy sqlite;

# activate package
echo "IMPORTANT - RUN: conda activate mip_v1";
