## Importing packages
import torch
import random
import torch.nn as nn 
import torchvision
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def gradient(image):
    '''
    Calculating the gradient using finite differences

    Parameters:
        image(torch.Tensor): The input image
    Returns:
        image(torch.Tensor): The gradient of the image in x and y respectively
    '''
    image = image.to(device)
    if len(image.shape) == 2:  # Add batch and channel dimensions if not present
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)

    # Define finite difference kernels
    # [[-1, 1]] 
    # [[-1], [1]]
    #[[0, 0, 0], [0, -1, 1], [0, 0, 0]]
    #[[0, 0, 0], [0, -1, 0], [0, 1, 0]]
    kernel_x = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    kernel_y = torch.tensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    grad_x = F.conv2d(image, kernel_x,padding=1).to(device)
    grad_y = F.conv2d(image, kernel_y,padding=1).to(device)
    return grad_x, grad_y
    



def divergence(p1, p2):
    '''
    Calculating the divergence using finite differences

    Parameters:
        image(torch.Tensor): The input image
    Returns:
        image(torch.Tensor): The divergence of the image in x and y respectively
    '''
    p1 = p1.to(device)
    p2 = p2.to(device)

    kernel_x = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    kernel_y = torch.tensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    div_p1 = F.conv_transpose2d(p1, kernel_x,padding=1).to(device)
    div_p2 = F.conv_transpose2d(p2, kernel_y,padding=1).to(device)
    return div_p1 + div_p2




#chambolle pock algorithm 2
def chambolle_pock_2(image,lam,tau_n,sigma_n,theta_n,gamma, number_iter,y1,y2):
    """
    Chambolle pock algorithm for ROF denoising
    
    Input Parameters:
        image (torch.Tensor): The input image
        number_iter(int): Number of times to repeat the algorithm above
        lam (float): Regularisation parameter 
    
    Returns:
        image (torch.Tensor): The denoised image
    """
    #loading into cuda
    image = image.to(device)
    y1 = y1.to(device)
    y2 = y2.to(device)

    #Regularising the image
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add batch dimension




    #Trying with set parameters tau sigma and theta for now
    #initialization
    x = image.clone()
    x_bar = x.clone()
    #Iterations and update
    for _ in range(number_iter):
        #Update the variables
        #updating the first equation
        # Update dual variables
        x_bar_grad_x, x_bar_grad_y = gradient(x_bar)
        y1 = y1 + sigma_n*x_bar_grad_x
        y2 = y2 + sigma_n*x_bar_grad_y
        norm = torch.sqrt(y1 ** 2 + y2 ** 2).clamp(min=1)
        y1_new = y1 / norm
        y2_new = y2 / norm
        #updating second equation
        div_y = divergence(y1_new,y2_new)
        x_old = x.clone()
        x = (x + tau_n * lam * (image) - tau_n*(div_y)) / (1 + tau_n * lam)
        #updating theta
        theta_n = 1/(torch.sqrt(torch.tensor(1+2*gamma*tau_n))).to(device)
        tau_n = theta_n*tau_n
        sigma_n = sigma_n/theta_n
        #updating x_bar
        x_bar = x + theta_n*(x-x_old)
        #stopping crieterion
        stopping_criterion = 1/2 * ( torch.norm(x - x_old) /torch.norm(x))
        if stopping_criterion < 0.00001:
            return x,y1_new,y2_new
    return x,y1_new,y2_new


def spectral(N,image,lam=7,tau=0.08,sigma=1.0/(7*0.08),theta=1.0,gamma=0.35*7):
    '''
    Input parameters:
        N (integer): The number of spectral components computed
        image (torch.tensor): The input image
        lam (float): Regularisiation parameter
        tau,sigma,theta,gamma (float): Parameters for the algorithm
    Output parameters:
        image (torch.tensor): The input image
        w_N (torch.tensor): The augmented image
        u_n (torch.tensor): The iterations of the TV denoising u(n)
    '''
    #loading into cuda
    image = image.to(device)
    batch,channel,height,width = image.shape
    #initialisng the u(n) vector
    u_n = torch.zeros((N+1,batch,channel,height,width)).to(device)
    #u(0) is always the image
    u_n[0] = image
    y1 = torch.zeros_like(image).to(device)
    y2 = torch.zeros_like(image).to(device)
    #loop to compute  and store u(n) for all n
    for i in range(N):
        u_n[i+1],y1,y2 = chambolle_pock_2(u_n[i],lam,tau,sigma,theta,gamma,5000,y1,y2)
    #calculating the augmented image
    w_N = phi_sum(N-2,u_n)  + desired_spectral_components(1,3,u_n) + f_r(N-2,u_n) ##(1st and 3rd components for CIFAR)
    return image,w_N,u_n


def phi(n,u_n,image_number = None):
    '''
    Input parameters:
        n (integer): The specific spectral component
        u_n (torch.tensor): The iterations of the TV denoising u(n)
        image_number (integer): The specific components we wish to amplify
    Output parameter:
        phi(n) (torch.tensor): The phi component at time step n
    '''
    #calculates phi in the paper
    if image_number != None:
        return n*(u_n[n-1,image_number] + u_n[n+1,image_number]-2*u_n[n,image_number]) 
    return n*(u_n[n-1] + u_n[n+1]-2*u_n[n]) 

def phi_sum(N,u_n,image_number = None):
    '''
    Input parameters:
        n (integer): The number of spectral components computed
        u_n (torch.tensor): The iterations of the TV denoising u(n)
        image_number (integer): The specific components we wish to amplify
    Output parameter:
        Sum of all phi(n) (torch.tensor): The sum of all the phi components at final time step
    '''
    #calculates the sum of all the phi components 
    if image_number != None:
        return u_n[0,image_number] - (N+1)*u_n[N,image_number] + N*u_n[N+1,image_number]
    return u_n[0] - (N+1)*u_n[N] + N*u_n[N+1]

def f_r(n,u_n,image_number = None):
    '''
    Input parameters:
        n (integer): The number of spectral components computed
        u_n (torch.tensor): The iterations of the TV denoising u(n)
        image_number (integer): The specific components we wish to amplify
    Output parameter:
        f_r(n) (torch.tensor): The residual at time step n
    '''
    if image_number != None:
        return (n+1)*u_n[n,image_number] - n*u_n[n+1,image_number]
    return (n+1)*u_n[n] - n*u_n[n+1]

def S(n,u_n):
    '''
    Input parameters:
        n (integer): The specific spectral component
        u_n (torch.tensor): The iterations of the TV denoising u(n)
        image_number (integer): The specific components we wish to amplify
    Output parameter:
        S(n) (torch.tensor): The spectral information at time step n
    '''
    return torch.sum(torch.abs(phi(n,u_n)))


def desired_spectral_components(start,end,u_n):
    '''
    Input parameters:
        [start,end) (integers): The range of values we wish to amplify (end not included)
    Output:
        phi([start,end)) (torch.tensor): The range of values of phi that were amplified 
    '''
    phi_old = 0
    phi_new = 0
    for j in range(start,end):
        phi_old += phi(j,u_n)
        #phi_new += coeff[j-start]*phi(j,u_n)
        phi_new += random.randrange(3,5)*phi(j,u_n)
    return phi_new - phi_old


'''
Functions below not used but included for completion
'''


def desired_spectral_components_gan(start,end,u_n,image_number):
    '''
    Put range of values you would like to change
    start - end(not included)
    '''
    phi_old = 0
    phi_new = 0
    spec,batch,channel,height,width = u_n.shape
    for j in range(start,end):
        phi_old += phi(j,u_n,image_number)
        phi_new += random.randrange(1,2)*phi(j,u_n,random.randrange(0,batch))
    return phi_new - phi_old

def combine_spectral_components(N,w_N,u_n):
    '''
    N is the number of spectral components computed
    u_n is the computed spectral list
    image_number is the chosen image in the batch we want to change
    '''
    spec,batch,channel,height,width = u_n.shape
    for i in range(batch):
        w_N[i] = phi_sum(N-2,u_n,i) + f_r(N-2,u_n,i) + desired_spectral_components_gan(7,8,u_n,image_number=i)
    return w_N
