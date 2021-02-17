#!/bin/bash

source /opt/anaconda3/etc/profile.d/conda.sh;
conda activate mip_v1 && python mip/app.py "$@"
