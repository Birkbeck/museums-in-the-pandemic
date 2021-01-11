# list all conda environments
conda env list;

# export environment to yml

conda env export -n mip_v1 --from-history | grep -v "prefix" > mip_v1.yml;

conda env export -n mip_v1 | grep -v "prefix" > mip_v1_full.yml;
