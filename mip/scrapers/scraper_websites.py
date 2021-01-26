# -*- coding: utf-8 -*-
#
# Web scraper

import time
import uuid
import json
import os
import random
import calendar
import datetime
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from db.db import *

def scrape_websites(museums_df):
    """ """
    print("scrape_websites", len(museums_df))
    
