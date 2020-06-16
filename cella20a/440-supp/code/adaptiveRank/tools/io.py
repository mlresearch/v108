import numpy as np

PRINT_THRESHOLD = 2 

def c_print(score, string):
    if score > PRINT_THRESHOLD:
        print(string)

def np_save(path, obj):
    with open(path, 'w') as f:
        np.save(f, obj)
        f.close()

def np_load(path):
    with open(path, 'r') as f:
        return np.load(f)
