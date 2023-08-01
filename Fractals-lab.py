import numpy as np
import torch
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Part1(device):
    """This part is for the code involved in part 1 of the lab work"""
    # grid for computing image, subdivide the space
    X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

    # load into PyTorch tensors
    x = torch.Tensor(X)
    y = torch.Tensor(Y)

    # transfer to the GPU device
    x = x.to(device)
    y = y.to(device)

    # Compute Gaussian
    z = torch.exp(-(x**2+y**2)/2.0)
    sin = torch.sin(2*np.pi*(4*x + 4*y))
    Guass_Sin = z*sin


    #plot
    plt.imshow(Guass_Sin.numpy())
    plt.tight_layout()
    plt.show()

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

def Part2(device):
    # Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
    Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]

    # load into PyTorch tensors
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    z = torch.complex(x, y) #important!
    zs = torch.zeros_like(z)
    ns = torch.zeros_like(z)

    # transfer to the GPU device
    z = z.to(device)
    zs = zs.to(device)
    ns = ns.to(device)

    #Mandelbrot Set
    for i in range(200):
        #Compute the new values of z: z^2 + x
        zs_ = zs*zs + z
        #Have we diverged with this new value?
        not_diverged = torch.abs(zs_) < 4.0
        #Update variables to compute
        ns += not_diverged.type(torch.FloatTensor)
        zs = zs_

    fig = plt.figure(figsize=(16,10))
    plt.imshow(processFractal(ns.numpy()))
    plt.tight_layout(pad=0)
    plt.show()

def main():
    Part1(device)
    #Part2(device)

if __name__ == '__main__': main()