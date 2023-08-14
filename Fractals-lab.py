"""
This is for creating and plotting an Ikeda map
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

def plot_ikeda(x_total, y_total, max_iterations, num_points):
    """Plots the ikeda map to matplotlib
    x_total: tensor dim 0 of all x values
    y_total: tensor dim 0 of all y values
    """
    #move to cpu
    x_total = x_total.cpu()
    y_total = y_total.cpu()

    #convert to arrays
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
    for i in range(num_points): #Add values for x, y into lists
        p_x = random.uniform(-r, r)
        list_x.append(p_x)
        p_y = random.uniform(-r, r)
        list_y.append(p_y)
    return list_x,list_y

def ikeda_map(u, list_x, list_y, max_iterations, device):
    """Creates 
    """
    #setup variables
    x = torch.FloatTensor(list_x)
    y = torch.FloatTensor(list_y)
    t_n = torch.zeros_like(x)
    x_ = torch.zeros_like(x)
    y_ = torch.zeros_like(y)
    x_total = torch.FloatTensor()
    y_total = torch.FloatTensor()

    #Move variables to gpu (or device is cpu)
    x = x.to(device)
    y = y.to(device)
    t_n = t_n.to(device)
    x_total = x_total.to(device)
    y_total = y_total.to(device)
    x_ = x_.to(device)
    y_ = y_.to(device)

    #Calculate the new x,y pos per iteration
    for i in range(max_iterations):
        #calualte the value of t_n
        t_n = 0.4 -(6/(1 + x**2 + y**2))
        #calculate new x and y pos
        x_ = 1 + u*(x*torch.cos(t_n)-y*torch.sin(t_n))
        y_ = u*(x*torch.sin(t_n)+y*torch.cos(t_n))
        #Combine with all prevous iterations
        x_total =  torch.cat((x_total, x_),0)
        y_total = torch.cat((y_total, y_),0)
        x = x_
        y = y_
    return x_total, y_total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Set up varibles
    u = 0.95
    r = 5
    max_iterations = 500
    num_points = 20000

    #calculate and plot ikeda map
    x,y = inti_points(num_points, r)
    x_total, y_total = ikeda_map(u, x, y, max_iterations, device)
    plot_ikeda(x_total, y_total, max_iterations, num_points)

if __name__ == '__main__': main()
