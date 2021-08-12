#!/usr/bin/env python
# coding: utf-8

#Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import kuibit
import kuibit.simdir as sd
from kuibit import grid_data as gd
from kuibit import grid_data_utils as gdu
from kuibit import series
from matplotlib.colors import LogNorm, Normalize
from matplotlib import cm
import h5py
from scipy.interpolate import griddata
from tqdm import tqdm
import xarray as xr
import matplotlib.colors as colors
import warnings
import argparse
import os
warnings.filterwarnings("ignore")

#Argparse stuff
parser = argparse.ArgumentParser()
parser.add_argument("--data",type=str, help="Path to file with simulation data", required=True)
parser.add_argument("--eos",type=str, help="Path to EoS file", required=True)
parser.add_argument("--imgdir",type=str, help="Path to where the individual slice images would be stored", required=True)
parser.add_argument("--outdir",type=str, help="Desired name of the output video file and the .h5 datafile", required=True)
parser.add_argument("--or_x",type=float, help="Desired value for the origin of the x axis on the grid", required=False, default=-10)
parser.add_argument("--or_y",type=float, help="Desired value for the origin of the y axis on the grid", required=False, default=-10)
parser.add_argument("--dx",type=float, help="Desired value for the origin of the x axis on the grid", required=False, default=0.4)
parser.add_argument("--dy",type=float, help="Desired value for the origin of the y axis on the grid", required=False, default=0.4)
parser.add_argument("--vidout",type=bool, help="Do you want a video simulation of the Schwarzschild data?", required=False, default=True)
parser.add_argument("--verbose",type=bool, help="Blabber", required=False, default=False)

args = parser.parse_args()


#Importing directories
if(args.verbose==True):
    print('Setting input parameters')
sdir = args.data
eos_file=h5py.File(args.eos,'r')
sim = sd.SimDir(sdir)

#Importing functions

funcnames = {'rho.xy': sim.gf.xy['rho'],'temperature.xy': sim.gf.xy['temperature'],
             'press.xy': sim.gf.xy['press'],'eps.xy': sim.gf.xy['eps'],'Y_e.xy': sim.gf.xy['Y_e']}


#Grid parameters required
origin_x = args.or_x
dx = args.dx
x = np.arange(origin_x,-origin_x+dx,dx)
nx = x.shape[0]

origin_y = args.or_y
dy = args.dy
y = np.arange(origin_y,-origin_y+dy,dy)
ny = y.shape[0]

n = nx


outfile = args.outdir


# # Creating modified grids as required 

if(args.verbose==True):
    print('Making grids based on input params...')
grid = gd.UniformGrid([nx, ny], # Number of points
                      x0=[origin_x, origin_y], # origin
                      x1=[-origin_x, -origin_y] # other corner
                      )

         
rho_dg = sim.gf.xy['rho']
T_dg = sim.gf.xy['temperature']
p_dg = sim.gf.xy['press']
eps_dg = sim.gf.xy['eps']
ye_dg = sim.gf.xy['Y_e']
W_dg = sim.gf.xy['w_lorentz']



#Creating kuibit .h5 files


if(args.verbose==True):
    print('Creating kuibit custom files...')
for key in funcnames:
    for i in funcnames[key].iterations:
        dg = funcnames[key].get_iteration(i).to_UniformGridData_from_grid(grid).data_xyz
        df = h5py.File(key+'.h5', 'a')
        df.create_dataset(str(i), data=dg)
        df.close()
    if(args.verbose==True):
        print(key+'.h5 is created')

# Using modified .h5 files for sch file creation

# In[110]:

if(args.verbose==True):
    print('Starting Schwarzschild discriminant calculations...')
#Importing files
slice_ye=h5py.File('Y_e.xy.h5','r')
slice_temp=h5py.File('temperature.xy.h5','r')
slice_rho=h5py.File('rho.xy.h5','r')
slice_eps=h5py.File('eps.xy.h5','r')
slice_press=h5py.File('press.xy.h5','r')
ref_file=h5py.File(sdir+'/output-0000/hdf5_2D/rho.xy.h5','r')


keys_rho = ref_file.keys()
keys_rho = list(keys_rho)



#Defining Constants
len_c = 6.77269222552442e-06
time_c = 2.03040204956746e05
rho_c = 1.61887093132742e-18
press_c =  1.80123683248503e-39
eps_c = 1.11265005605362e-21

#gr = np.array(slice_press['Parameters and Global Attributes']['Grid Structure v5'])
gr = np.array(ref_file[keys_rho[-1]]['Grid Structure v5'])

#Assuming all logs from the simulation to be natural and from the EoS to be base10

#Velocity of Sound squared
cs2 = np.array(eos_file['cs2'])  
#Log eps
loge = np.array((eos_file['logenergy']))
#Log Pressure
logp = np.array((eos_file['logpress']))
#Log Rho
logrho = np.array(eos_file['logrho'])
#Log temp
logtemp = np.array(eos_file['logtemp'])
#Eps
e = 10**loge
#Pressure
p = 10**logp
#Rho
rho = 10**(np.array(eos_file['logrho']))
#Temperature 
temp = 10**(np.array(eos_file['logtemp']))
#Ye
ye = np.array(eos_file['ye'])
#Energy_shift
esh = np.array(eos_file['energy_shift'])

for keyno in tqdm(rho_dg.iterations):
#for keyno in tqdm(range(0,1)):
    keyno = str(keyno)
    #print('Starting with the slice',format(keys_rho[keyno]))
    #Slice_rho
    rho_s = (np.array(slice_rho[keyno]))/rho_c
    #Slice_temp
    temp_s = np.array(slice_temp[keyno])
    #Slice_ye
    ye_s = np.array(slice_ye[keyno])
    #Slice_eps
    eps_s = ((np.array(slice_eps[keyno]))/eps_c) + esh
    #Slice_press
    press_s = (np.array(slice_press[keyno]))/press_c


    #Defining gridpoints for interpolation according to 1D requirement
    flat_ye = ye_s.flatten()
    flat_temp = temp_s.flatten()
    flat_temp = np.maximum(flat_temp,0.101*np.ones_like(flat_temp))
    flat_rho = rho_s.flatten()
    grid_final=[]
    da = xr.DataArray(cs2,coords=[ye,temp,rho],dims=["ye","temp","rho"])
    ye_da = xr.DataArray(flat_ye,dims="z")
    temp_da = xr.DataArray(flat_temp,dims="z")
    rho_da = xr.DataArray(flat_rho,dims="z")
    grid_final = np.array(da.interp(ye=ye_da,temp=temp_da,rho=rho_da))
    grid_final = grid_final.reshape(ye_s.shape)

    #Calculation of the Schwarzschild Discriminant 

    A = []

    A = np.zeros_like(eps_s)

    for i in range(n):
        for j in range(0,n):
            if i == 0:
                diffe_x = (eps_s[1,j]-eps_s[0,j])/dx
                diffp_x = (press_s[1,j]-press_s[0,j])/dx
            elif i == n-1:
                diffe_x = (eps_s[n-1,j]-eps_s[n-2,j])/dx
                diffp_x = (press_s[n-1,j]-press_s[n-2,j])/dx
            else:
                diffe_x = (eps_s[i+1,j]-eps_s[i-1,j])/(2*dx)
                diffp_x = (press_s[i+1,j]-press_s[i-1,j])/(2*dx)

            if j == 0:
                diffe_y = (eps_s[i,1]-eps_s[i,0])/dy
                diffp_y = (press_s[i,1]-press_s[i,0])/dy
            elif j == n-1:
                diffe_y = (eps_s[i,n-1]-eps_s[i,n-2])/dy
                diffp_y = (press_s[i,n-1]-press_s[i,n-2])/dy
            else:
                diffe_y = (eps_s[i,j+1]-eps_s[i,j-1])/(2*dy)
                diffp_y = (press_s[i,j+1]-press_s[i,j-1])/(2*dy)



            #Gamma Calculation 
            Gamma = (((grid_final[i,j])*(eps_s[i,j] + press_s[i,j]))/press_s[i,j])

            A_x = 2*((1/(eps_s[i,j] + press_s[i,j]))*diffe_x  - (1/(Gamma*press_s[i,j]))*diffp_x)*len_c
            A_y = 2*((1/(eps_s[i,j] + press_s[i,j]))*diffe_y  - (1/(Gamma*press_s[i,j]))*diffp_y)*len_c

            theta = np.arctan2(y[j], x[i])
            A[i, j] = np.cos(theta) * A_x + np.sin(theta) * A_y
            #A[i,j] = A_x + A_y

    sc = h5py.File(outfile+'.h5', 'a')
    sc.create_dataset(str(keyno), data=A)
    sc.close()


# # Post processing

if(args.verbose==True):
    print('Starting post-processing')
final_file = h5py.File(outfile+'.h5', 'r')


for i in tqdm(rho_dg.iterations):
    i = str(i)
    x = np.arange(origin_x, -origin_x+dx, dx)       
    y = np.arange(origin_y, -origin_y+dx, dx)
    X, Y = np.meshgrid(x, y)
    plt.pcolormesh(X, Y, np.array(final_file[i]),norm=colors.SymLogNorm(linthresh=1e-13, linscale=1,
                                          vmin=-1e-9, vmax=1e-9, base=10),cmap=plt.get_cmap('seismic'))
    plt.colorbar()
    plt.contour(X,Y,slice_rho[i],levels=[1e-9],linewidths=3,colors=['green'])

    plt.title('Schwarzschild Discriminant Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(args.imgdir + i + '.png')
    plt.close()
final_file.close()

for key in funcnames:
    os.remove(key+".h5")

if(args.vidout==True):
    if(args.verbose==True):
        print('Making a video..almost done')
    import os
    import moviepy.video.io.ImageSequenceClip
    image_folder='Images'
    fps=1.2 
    images=[]
    for i in range(0,16640+128,128):
        images.append(args.imgdir + str(i) + '.png')
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=fps)
    clip.write_videofile(outfile+'.mp4')
