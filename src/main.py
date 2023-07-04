import os
import sys


def main():
    if len(sys.argv) != 2:
        print("usage: plotgen [file]")
        return

    path = os.path.join(os.path.split(__file__)[0], "plotting.py")
    filename = sys.argv[1]

    if not os.path.exists(filename):
        print(f"plotgen: {filename}: No such file or directory")
        return
    if os.path.isdir(filename):
        print(f"plotgen: {filename}: Is a directory")
        return

    assert os.path.exists(path)  # this should be correct regardless of input

    os.system(f"bokeh serve {path} --args {filename}")
