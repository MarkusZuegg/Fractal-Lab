"""
This is for creating and plotting an Ikeda map
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

def plot_ikeda(x_total, y_total, max_iterations, num_points):
    """Plots the ikeda map to matplotlib
    x_total: tensor dim 0 of all x values
    y_total: tensor dim 0 of all y values
    """
    x_array = x_total.numpy()
    y_array = y_total.numpy()
    fig = plt.figure(figsize=(16,10))
    plt.plot(x_array, y_array, 'bo', markersize=0.01)
    plt.tight_layout(pad=0)
    plt.show()

def inti_points(num_points, r):
    """Create two tensors with all the x values and y values of points
        num_points: int
        r: int of the range of x and y
        returns two lists of x and y values"""
    list_x = []
    list_y = []
    for i in range(num_points):
        p_x = random.uniform(-r, r)
        list_x.append(p_x)
        p_y = random.uniform(-r, r)
        list_y.append(p_y)
    return list_x,list_y

def ikeda_map(u, list_x, list_y, max_iterations, device):
    """"""
    #setup variables
    x = torch.FloatTensor(list_x)
    y = torch.FloatTensor(list_y)
    t_n = torch.zeros_like(x)
    x_total = torch.Tensor()
    y_total = torch.Tensor()

    #Move variables to gpu
    x = x.to(device)
    y = y.to(device)
    t_n = t_n.to(device)
    x_total = x_total.to(device)
    y_total = y_total.to(device)

    #Calculate the new x,y pos per iteration
    for i in range(max_iterations):
        t_n = 0.4 -(6/(1 + x**2 + y**2))
        x_ = 1 + u*(x*np.cos(t_n)-y*np.sin(t_n))
        y_ = u*(x*np.sin(t_n)+y*np.cos(t_n))
        x_total =  torch.cat((x_total, x_),0)
        y_total = torch.cat((y_total, y_),0)
        x = x_
        y = y_
    print(x_total)
    print(y_total)
    return x_total, y_total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    u = 0.95
    r = 5
    max_iterations = 500
    num_points = 20000
    x,y = inti_points(num_points, r)
    x_total, y_total = ikeda_map(u, x, y, max_iterations, device)
    plot_ikeda(x_total, y_total, max_iterations, num_points)


if __name__ == '__main__': main()
