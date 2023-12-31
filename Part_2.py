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
    R_ = torch.linspace(r_l,r_u, res)
    I_ = torch.linspace(i_l,i_u, res)
    I, R = torch.meshgrid(I_, R_) # creates tensors as output
    return I, R

def draw_fractal(array):
    """Draws Fractal to matplotlib"""
    fig = plt.figure(figsize=(16,10))
    plt.imshow(processFractal(array.cpu().numpy()))
    plt.tight_layout(pad=0)
    plt.show()

def Point_create_mgrid(c_real, c_imag, width, res):
    """Creates meshgrid centred around given complex point"""
    scale = width/2
    r_l = c_real-scale
    r_u = c_real+scale
    i_l = c_imag-scale
    i_u = c_imag+scale
    I, R = create_mgrid(i_l, i_u, r_l, r_u, res)
    return I, R


def Gen_Mandelbrot(device, R, I, max_iterations):
    """This function generates a Mandelbrot set
    Given a 2D real number array and a 2D imaginary array of same size
    And returns a Torch array
    """
    z = torch.complex(R, I) #combines both real, imaginary grids into complex number grid
    zs = z.clone()
    ns = torch.zeros_like(z) 
    not_diverged = torch.zeros_like(z)

    # transfer to the GPU device
    z = z.to(device)
    zs = zs.to(device)
    ns = ns.to(device)
    not_diverged = not_diverged.to(device)

    #Mandelbrot Set
    for i in range(max_iterations):
        #Compute the new values of z: z^2 + x
        zs_ = zs*zs + z
        #Have we diverged with this new value?
        not_diverged = torch.abs(zs_) < 4.0
        #Update variables to compute
        ns += not_diverged
        zs = zs_
    return ns

def Gen_Julia_set(device, R, I, c_real, c_imag, max_iterations):
    """This function generates a Julia set
    Given a 2D real number array and a 2D imaginary array of same size
    And a complex constant c (c_real + i*c_imag)
    And returns a Torch array
    """

    z = torch.complex(R, I) #combines both real, imaginary grids into complex number grid
    zs = z.clone()
    ns = torch.zeros_like(z)
    not_diverged = torch.zeros_like(z)
    c = torch.complex(torch.tensor(c_real), torch.tensor(c_imag))

    # transfer to the GPU device
    z = z.to(device)
    zs = zs.to(device)
    ns = ns.to(device)
    not_diverged = not_diverged.to(device)
    c = c.to(device)

    for i in range(max_iterations):
        #Compute the new values of z: z^2 + x
        zs_ = zs*zs + c
        #Have we diverged with this new value?
        not_diverged = torch.abs(zs_) < 4.0
        #Update variables to compute
        ns += not_diverged
        zs = zs_
    return ns


def main():
    #choose device for pytorch to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Resultion for all
    res = 1000

    #Mandelbrot setup
    r_upper = 1 # setup grid of real numbers
    r_lower = -2
    i_upper = 1.3 # setup grid for imaginary numbers
    i_lower = -1.3
    I_xy, R_xy = create_mgrid(i_lower, i_upper, r_lower, r_upper, res)
    #create and draw fractal
    Mandelbrot = Gen_Mandelbrot(device, R_xy, I_xy, 500)
    draw_fractal(Mandelbrot)

    #Zooming in on mandlebrot
    p_real = -0.759856#-0.87591
    p_imag = 0.125547#0.20464
    width = 0.051
    zoom_I, zoom_R = Point_create_mgrid(p_real, p_imag, width, res) 
    Mandle_zoom = Gen_Mandelbrot(device, zoom_R, zoom_I, 700)
    draw_fractal(Mandle_zoom)

    #Julia set
    c_real = -0.744
    c_imag = 0.148
    I , R = create_mgrid(-2,2,-2,2,res)
    julia = Gen_Julia_set(device, R,I, c_real, c_imag, 500)
    draw_fractal(julia)

if __name__ == '__main__': main()