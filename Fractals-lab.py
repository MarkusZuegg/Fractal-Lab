"""
This is for the 3rd part. Creating and coding my own fractal set
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

def processFractal(a):
    """Display an array of iteration counts as a colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic), 30+50*np.sin(a_cyclic), 155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

def create_mgrid(i_l, i_u, r_l, r_u, res):
    """creates two 2D arrays given upper and lower limits"""
    # modify to use linspace for accurate res
    R_ = torch.linspace(r_l,r_u, res)
    I_ = torch.linspace(i_l,i_u, res)
    I, R = torch.meshgrid(I_, R_)
    return I, R

def draw_fractal(array):
    fig = plt.figure(figsize=(16,10))
    plt.imshow(processFractal(array.numpy()))
    plt.tight_layout(pad=0)
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return

if __name__ == '__main__': main()
