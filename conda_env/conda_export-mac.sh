# list all conda environments
conda env list;

# export environment to yml

conda env export -n mip_v1 --from-history | grep -v "prefix" > mip_v1.yml;
# works for regular pip packages
echo "  - pip" >> mip_v1.yml;
echo "  - pip:" >> mip_v1.yml;
echo "    - webbot" >> mip_v1.yml;

conda env export -n mip_v1 | grep -v "prefix" > mip_v1_full_mac.yml;
