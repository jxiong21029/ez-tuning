import os
import sys


def main():
    path = os.path.join(os.path.split(__file__)[0], "plotting.py")
    os.system(f"bokeh serve {path} --args {sys.argv[1]}")
