"""
Utility for model
"""
import pathlib
import os
import json

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def mkdir_p(full_dir):
    """Simulate mkdir -p"""
    if not os.path.exists(full_dir):
        pathlib.Path(full_dir).mkdir(parents=True, exist_ok=True)
