import numpy as np
from PyPIC3D.utilities.grids import build_collocated_axis, build_staggered_axis

x_wind = 1
y_wind = 1
nx = 200
ny = 200
dx = x_wind / nx
dy = y_wind / ny
# spatial resolution

x = np.arange(0, x_wind, dx)
y = np.arange(0, y_wind, dy)
X, Y = np.meshgrid(x, y)
# define the grid

vertex_x = build_staggered_axis(0, dx, nx)[1:-1]
vertex_y = build_staggered_axis(0, dy, ny)[1:-1]
# build the staggered axes for the vertex grid

center_x = build_collocated_axis(0, dx, nx)[1:-1]
center_y = build_collocated_axis(0, dy, ny)[1:-1]
# build the collocated axes for the center grid

# Ex_grid = vertex_x, center_y, center_z
# Ey_grid = center_x, vertex_y, center_z
# Ez_grid = center_x, center_y, vertex_z
# Bx_grid = center_x, vertex_y, vertex_z
# By_grid = vertex_x, center_y, vertex_z
# Bz_grid = vertex_x, vertex_y, center_z
# from PyPIC3D staggered grid conventions

Bx_X, Bx_Y = np.meshgrid(center_x, vertex_y, indexing='ij')
By_X, By_Y = np.meshgrid(vertex_x, center_y, indexing='ij')
# define the grids for the magnetic field components

B0 = 0.01  #1 / np.sqrt(4 * np.pi)
# magnetic field strength

C  = 2.9e8
V0 = 1.0 * C
# velocity magnitude

Bx = -B0 * np.sin(2 * np.pi * Bx_Y / y_wind)
By = B0 * np.sin(4 * np.pi * By_X / x_wind)
# components of the magnetic field

number_density = 1e14
PPC = 15
N_particles = int(PPC * nx * ny)
# number of particles

electron_x = np.random.uniform(0, x_wind, N_particles)
electron_y = np.random.uniform(0, y_wind, N_particles)
electron_z = np.zeros(N_particles)
# initial positions of electrons

ion_x = electron_x.copy()
ion_y = electron_y.copy()
ion_z = electron_z.copy()
# initial positions of ions

electron_vx = -V0  * np.sin(2* np.pi * electron_y / y_wind)
electron_vy =  V0  * np.sin(2* np.pi * electron_x / x_wind)
electron_vz = np.zeros(N_particles)
# initial velocities of electrons

ion_vx = electron_vx.copy()
ion_vy = electron_vy.copy()
ion_vz = electron_vz.copy()
# initial velocities of electrons and ions

Bx = np.expand_dims(Bx, axis=-1)
By = np.expand_dims(By, axis=-1)
# Expand dimensions to match fields shape


electron_u = np.sqrt(electron_vx**2 + electron_vy**2 + electron_vz**2)
ion_u = np.sqrt(ion_vx**2 + ion_vy**2 + ion_vz**2)

gamma_e = np.sqrt(1 + (electron_u / C)**2)
gamma_i = np.sqrt(1 + (ion_u / C)**2)
# compute the Lorentz factor for electrons and ions

electron_vx = electron_vx / gamma_e
electron_vy = electron_vy / gamma_e
ion_vx = ion_vx / gamma_i
ion_vy = ion_vy / gamma_i
# correct the velocities of electrons and ions for relativistic effects

electron_v = np.sqrt(electron_vx**2 + electron_vy**2 + electron_vz**2)
ion_v = np.sqrt(ion_vx**2 + ion_vy**2 + ion_vz**2)


mean_electron_v = np.mean(electron_v)
mean_ion_v = np.mean(ion_v)
# compute the mean velocities of electrons and ions for reporting

print(f"average electron velocity: {mean_electron_v / C} C")
print(f"average ion velocity: {mean_ion_v / C} C")

vth_e = np.sqrt( np.mean(electron_v**2) )
# compute the thermal velocity of electrons

mu0 = 1.25663706212e-6
q_e = 1.602176634e-19
me    = 9.10938356e-31
ep0   = 8.854187817e-12
# fundamental constants

kbT = 1/3 * me * vth_e**2
# compute the thermal energy of electrons in Joules

debye_length = np.sqrt(ep0 * kbT / (number_density * q_e**2))
# compute the Debye length in meters
print(f"Debye length: {debye_length} m")

dV             = dx * dy * debye_length
# number density and volume element
weight = number_density * dV / PPC
# calculate the weight of each particle
print(f"Number of particles: {N_particles}")
print(f"Weight of each particle: {weight}")
print(f" Grid points (debye lengths): {dx/debye_length} debye lengths, {dy/debye_length} debye lengths")


electron_x = electron_x - x_wind/2
electron_y = electron_y - y_wind/2
ion_x = ion_x - x_wind/2
ion_y = ion_y - y_wind/2
# shift the particle positions to be centered around (0,0)

np.save('Bx.npy', Bx)
np.save('By.npy', By)
np.save('electron_x.npy', electron_x)
np.save('electron_y.npy', electron_y)
np.save('electron_z.npy', electron_z)
np.save('ion_x.npy', ion_x)
np.save('ion_y.npy', ion_y)
np.save('ion_z.npy', ion_z)
np.save('electron_vx.npy', electron_vx)
np.save('electron_vy.npy', electron_vy)
np.save('ion_vx.npy', ion_vx)
np.save('ion_vy.npy', ion_vy)
np.save('electron_vz.npy', electron_vz)
np.save('ion_vz.npy', ion_vz)
# save npy arrays