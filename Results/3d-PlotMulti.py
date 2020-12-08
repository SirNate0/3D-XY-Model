#!/bin/python3
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,8)
import matplotlib.pyplot as plt
import numpy as np

for dir in ('3D Ising Metropolis','3D Ising Wolff'):
    label = dir
    if len(dir):
        dir += '/'
    energies = np.fromfile(dir+'energy.dat',dtype=np.double)
    mags = np.fromfile(dir+'magnetization.dat',dtype=np.double)
    dim,N,NT,steps,T0,Trange = np.genfromtxt(dir+'params.txt')
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


    label=label + ' N={}, steps={}'.format(N,steps)
    plt.figure(1)
    plt.subplot(221)
    plt.plot(T,avg_energy/N**dim,label=label)
    plt.xlabel("Temperature")
    plt.ylabel("Energy Density $E(T)$")
    plt.legend()
    plt.subplot(222)
    plt.plot(T,currentM_abs/N**dim,label=label)
    plt.xlabel("Temperature")
    plt.ylabel(r"Magnetization $|\vec{M}(T)|$")
    plt.legend()
    plt.subplot(223)
    plt.plot(T,C_T,label=label)
    plt.xlabel("Temperature")
    plt.ylabel("Specific Heat $U(T)$")
    plt.legend()
    plt.subplot(224)
    plt.plot(T,chi_T,label=label)
    plt.xlabel("Temperature")
    plt.ylabel("Susceptibility $\chi(T)$")
    plt.legend()
    plt.tight_layout()
    
    plt.figure(2)
    plt.title('Estimate Critical Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Binder cumulant')
    binder= 1-np.average(np.linalg.norm(mags[:,thermalize:],axis=2)**4,axis=1)/np.average(np.linalg.norm(mags[:,thermalize:],axis=2)**2,axis=1)**2
    plt.plot(T,binder,label=label)
    plt.legend()

    plt.figure()
    plt.subplot(211)
    #plt.plot(np.arange(steps)[::100],energies[:,::100].T)
    plt.plot(np.arange(steps)[::steps//400],energies[:,::steps//400].T)# Too hard to see: ,':')
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.title(label)
    plt.subplot(212)
    
    plt.plot(np.arange(steps)[::steps//400],np.linalg.norm(mags[:,::steps//400],axis=2).T)# Too hard to see: ,':')
    plt.xlabel("Iterations")
    plt.ylabel("|Magnetization|")

print("Plotted")
plt.show()
