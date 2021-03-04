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
	nohup ./run_app_server.sh scrape_websites > tmp/logs/nohup_${DATE}_log.txt 2>&1 &
	# nohup perl perl/run_experiment_parallel_ant.pl > nohup_perl_exp_logfile.${date} 2>&1 &
 	# echo ">>> Experiment running in background."

running:
	-@ps auxw | grep 'mip/app.py';
	#-@ps auxw | grep '[t]or';
	#-@netstat -ant | grep 9050;

logs:
	@ls -lh tmp/logs
	@tail tmp/logs/nohup*

wait:
	sleep 3;
