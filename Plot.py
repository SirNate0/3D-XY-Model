#!/bin/python3
import matplotlib.pyplot as plt
import numpy as np

energies = np.fromfile('energy.dat',dtype=np.double)
mags = np.fromfile('magnetization.dat',dtype=np.double)
dim,N,NT,steps,T0,Trange = np.genfromtxt('params.txt')
dim = int(dim)
N = int(N)
NT = int(NT)
steps = int(steps)
T = np.linspace(T0,T0+Trange,NT)
energies.shape = (NT,steps+1)
mags.shape = (NT,steps,-1)
vecDim = mags.shape[2]

beta = energies[:,-1]
energies = energies[:,:-1]

print(N,NT,steps)
print(mags)
print(energies)

thermalize = int(steps*0.9)


avg_energy = np.average(energies[:,thermalize:],axis=1)
currentM_abs = np.average(np.linalg.norm(mags[:,thermalize:],axis=2),axis=1)

chi_T = (np.average(np.linalg.norm(mags[:,thermalize:],axis=2)**2,axis=1)-currentM_abs**2)/(T*N**dim)
C_T = (np.average(energies[:,thermalize:]**2,axis=1)-avg_energy**2)/(T*N**dim)

#plt.plot([1,2],[3,4])
plt.subplot(211)
plt.plot(energies[:,::100].T)
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.subplot(212)
plt.plot(np.linalg.norm(mags[:,::100],axis=2).T)
plt.xlabel("Iterations")
plt.ylabel("|Magnetization|")
#plt.subplot(313)
#plt.plot((mags[0,::100,:]).T)
##plt.plot((mags[1,::100,1]).T,':')
#plt.xlabel("Iterations")
#plt.ylabel("Magnetization")
#plt.figure()
#plt.imshow(energies[:,::(steps//100)])
#plt.colorbar()

plt.figure()
plt.subplot(221)
plt.plot(T,avg_energy)
plt.xlabel("Temperature")
plt.ylabel("Energy $E(T)$")
plt.subplot(222)
plt.plot(T,currentM_abs)
plt.xlabel("Temperature")
plt.ylabel(r"Magnetization $|\vec{M}(T)|$")
plt.subplot(223)
plt.plot(T,C_T)
plt.xlabel("Temperature")
plt.ylabel("Specific Heat $U(T)$")
plt.subplot(224)
plt.plot(T,chi_T)
plt.xlabel("Temperature")
plt.ylabel("Susceptibility $\chi(T)$")
plt.tight_layout()

#plt.figure()
#plt.plot(T-1/beta)

print("Plotted")
plt.show()
