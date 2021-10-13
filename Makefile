#
# MAKEFILE for MiP
#

SHELL := /bin/bash

DATE=`date +"%Y%m%d_%H"`

all:
	@echo "Select an option.";
	@cat Makefile;
	@echo "\n";


scrape_websites:
	@echo ">>> Scraping websites with nohup";
	nohup ./run_app_server.sh scrape_websites > tmp/logs/nohup_${DATE}_log.txt &
	# nohup perl perl/run_experiment_parallel_ant.pl > nohup_perl_exp_logfile.${date} 2>&1 &
 	# echo ">>> Experiment running in background."


db_stats:
	@echo ">>> DB stats";
	./run_app_server.sh db_stats;

scrape_facebook:
	@echo ">>> Facebook";
	nohup ./run_app_server.sh scrape_facebook > tmp/logs/nohup_fb_${DATE}_log.txt &


scrape_twitter:
	@echo ">>> Twitter";
	nohup ./run_app_server.sh scrape_twitter > tmp/logs/nohup_tw_${DATE}_log.txt &


extract_txt_fields:
	@echo ">>> Extract text fields";
	nohup ./run_app_server.sh ex_txt_fields > tmp/logs/nohup_${DATE}_log.txt &


an_text:
	@echo ">>> Run NLP";
	nohup ./run_app_server.sh an_text > tmp/logs/nohup_an_${DATE}_log.txt &


compile:
	@python -m compileall mip/*


running:
	-@ps auxw | grep 'mip/app.py';
	#-@ps auxw | grep '[t]or';
	#-@netstat -ant | grep 9050;

logs:
	@ls -lh tmp/logs/*
	@tail tmp/logs/nohup*txt

errors:
	@grep -i error tmp/logs/nohup*txt 

wait:
	sleep 3;
