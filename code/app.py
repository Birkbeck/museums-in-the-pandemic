# -*- coding: utf-8 -*-

# MIP project

# %% Setup

import os
import sys
import pandas as pd
from utils import StopWatch
from settings import Settings

COMMANDS = ["help","tests"]
cmd = None
settings = Settings()

# %% Operations
def init_app():
    print("Init App")
    
def cleanup():
    print("Cleanup")

# %% Main
def main():
    sw = StopWatch("app")
    init_app()

    print("== MIP App ==")
    print("Parameters: " + str(sys.argv[1:]))
    print("N CPUs =", os.cpu_count())
    if len(sys.argv) < 2 or not sys.argv[1] in COMMANDS:
        raise RuntimeError("Invalid parameter. Valid parameters: " + str(COMMANDS))

    cmd = sys.argv[1]

    if cmd == "scrape":
        print("scrape")
        # TODO

    elif cmd == "tests":
        print("run tests")
        # TODO

    cleanup()
    sw.tick("OK")
    print("OK")

if __name__ == '__main__':
    main()
