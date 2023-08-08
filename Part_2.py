"""
Mandelbrot set and julia set, along with zooming in on the Mandelbrot set
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

def Gen_Mandelbrot(device, R, I):
    """This function generates a Mandelbrot set
    Given a 2D real number array and a 2D imaginary array of same size
    And returns a Torch array
    """

    # load into PyTorch tensors
    xy_real = torch.Tensor(R)
    xy_imag = torch.Tensor(I)
    z = torch.complex(xy_real, xy_imag) #combines both real, imaginary grids into complex number grid
    zs = z
    ns = torch.zeros_like(z)
    print(z.shape)

    # transfer to the GPU device
    z = z.to(device)
    zs = zs.to(device)
    ns = ns.to(device)


    #Mandelbrot Set
    for i in range(500):
        #Compute the new values of z: z^2 + x
        zs_ = zs*zs + z
        #Have we diverged with this new value?
        not_diverged = torch.abs(zs_) < 4.0
        #Update variables to compute
        ns += not_diverged.type(torch.FloatTensor)
        zs = zs_
    return ns

def Gen_Julia_set(device, R, I, c_real, c_imag):
    """This function generates a Julia set
    Given a 2D real number array and a 2D imaginary array of same size
    And a complex constant c (c_real + i*c_imag)
    And returns a Torch array
    """

    # load into PyTorch tensors
    xy_real = torch.Tensor(R)
    xy_imag = torch.Tensor(I)
    z = torch.complex(xy_real, xy_imag) #combines both real, imaginary grids into complex number grid
    zs = z
    ns = torch.zeros_like(z)

    print(z.shape)

    # transfer to the GPU device
    z = z.to(device)
    zs = zs.to(device)
    ns = ns.to(device)

    c = torch.complex(torch.tensor(c_real), torch.tensor(c_imag))


    for i in range(500):
        #Compute the new values of z: z^2 + x
        zs_ = zs*zs + c
        #Have we diverged with this new value?
        not_diverged = torch.abs(zs_) < 4.0
        #Update variables to compute
        ns += not_diverged.type(torch.FloatTensor)
        zs = zs_
    return ns


def main():
    #choose device for pytorch to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #stepsize for grid
    stepsize = 1000
    #setup grid of real numbers
    r_upper = 1
    r_lower = -2
    #setup grid for imaginary numbers
    i_upper = 1.3
    i_lower = -1.3

    I_xy, R_xy = create_mgrid(i_lower, i_upper, r_lower, r_upper, stepsize)

    #create and draw fractal
    Mandelbrot = Gen_Mandelbrot(device, R_xy, I_xy)
    draw_fractal(Mandelbrot)

    c_real = -0.744
    c_imag = 0.148

    I, R = create_mgrid(-2,2,-2,2,1000)
    julia = Gen_Julia_set(device, R,I, c_real, c_imag)
    draw_fractal(julia)

if __name__ == '__main__': main()