"""
This is for the 3rd part. Creating and coding my own fractal set
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

def processFractal(a):
    """Display an array of iteration counts as a colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return

if __name__ == '__main__': main()
