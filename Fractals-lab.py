"""
This is for the 3rd part. Creating and coding my own fractal set
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

def processFractal(a):
    """Display an array of iteration counts as a colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic), 30+50*np.sin(a_cyclic), 155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

def draw_fractal(array):
    fig = plt.figure(figsize=(16,10))
    plt.imshow(processFractal(array.numpy()))
    plt.tight_layout(pad=0)
    plt.show()

def inti_points(num_points, r):
    """Create two tensors with all the x values and y values of points
        num_points: int
        r: int of the range of x and y"""
    list_x = []
    list_y = []
    for i in range(num_points):
        p_x = random.uniform(-r, r)
        list_x.append(p_x)
        p_y = random.uniform(-r, r)
        list_y.append(p_y)
    return list_x,list_y

def ikeda_map(u, list_x, list_y, max_iterations, device):
    #setup variables
    x = torch.FloatTensor(list_x)
    y = torch.FloatTensor(list_y)
    t_n = torch.zeros_like(x)

    #Move variables to gpu
    x = x.to(device)
    y = y.to(device)
    t_n = t_n.to(device)

    for i in range(max_iterations):
        t_n = 0.4 -(6/(1 + x**2 + y**2))
        x_ = 1 + u*(x*np.cos(t_n)-y*np.sin(t_n))
        y_ = u*(x*np.sin(t_n)+y*np.cos(t_n))
        x = x_
        y = y_
    return

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    p = inti_points(5, 10)

if __name__ == '__main__': main()
