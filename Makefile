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
	@echo "Scraping websites...";
	./run_app_server.sh scrape_websites

running:
	-@ps auxw | grep 'run_app_server';
	#-@ps auxw | grep '[t]or';
	#-@netstat -ant | grep 9050;

logs:
	@ls -lh tmp/logs

wait:
	sleep 3;
