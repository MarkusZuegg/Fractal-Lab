"""
Code for answering and understanding part 1 of lab. Understanding pytoch and python.
"""
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

    #2D sine function
    f = 0.5
    x_s = 4
    y_s = 4
    sin = torch.sin(f*torch.pi*(x_s*x + y_s*y))

    #Combination of sine and Guass
    Guass_Sin = z*sin


    #plot
    plt.imshow(Guass_Sin.cpu().numpy())
    plt.tight_layout()
    plt.show()

def main():
    Part1(device)

if __name__ == '__main__': main()