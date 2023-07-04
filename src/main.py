import os
import sys


def main():
    os.system(f"bokeh serve src/plotting.py --args {sys.argv[1]}")
