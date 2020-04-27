# -*- coding: utf-8 
"""
Much of this code was taken directly from the examples for fitting a single 
order spectrum at 
https://starfish.readthedocs.io/en/latest/examples/single.html

This was modified to conduct a PHOENIX model fit of HD108477 (G5 II) using
Starfish software.

Fitting for best parameter values of Teff, logg, and Z.

Created on Thu Apr 14 12:02:45 2020

@author: Christian
"""
#%%

import numpy as np
from Starfish.grid_tools import download_PHOENIX_models
import matplotlib.pyplot as plt

#%%

# read in data from text file and store data in arrays
file = open("HD108477_H_IGRINS_spectrum.txt", 'r')

line_list = file.readlines()

lam = np.zeros(len(line_list)-1)
flux = np.zeros(len(line_list)-1)
err = np.zeros(len(line_list)-1)

j=0
for i in range(len(line_list)-1):
    split_line = line_list[i+1].split('|')
    lam[j] = split_line[0]
    flux[j] = split_line[1]
    err[j] = split_line[2]
    j+=1
 
file.close()
  

  
# conduct unit conversions to be compatable with PHOENIX library
lam_f = lam*10000 # um to angstroms
flux_f = flux*1e7 # W/m^2/um to erg/s/cm^2/cm
err_f = err*1e7 # W/m^2/um to erg/s/cm^2/cm

#%%

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#Truncate spectrum to speed up process
a = find_nearest(lam_f, 16600)
b = find_nearest(lam_f, 16700)
lam_f = lam_f[a:b]
flux_f = flux_f[a:b]
err_f = err_f[a:b]



#%%

plt.plot(lam_f, flux_f)


#%%

from Starfish.spectrum import Spectrum
data = Spectrum(lam_f, flux_f, sigmas=err_f, name="HD108477 IGRINS H Spectrum")
data.plot()


#%%

# Download Pheonix models
ranges = [[4500, 8000], [0.5, 2.0], [-0.5,0.5]]
download_PHOENIX_models(path="PHOENIX", ranges=ranges)

#%%

# Setup a grid interface with the models
from Starfish.grid_tools import PHOENIXGridInterfaceNoAlpha

grid = PHOENIXGridInterfaceNoAlpha(path="PHOENIX")

# Setup an HDF5 interface in order to allow much quicker reading and writing 
# than compared to loading FITS files over and over again.
from Starfish.grid_tools.instruments import IGRINS_H_custom
from Starfish.grid_tools import HDF5Creator


creator = HDF5Creator(
    grid, "IGRINS_grid.hdf5",instrument=IGRINS_H_custom(), 
    wl_range=(16600, 16700), ranges=ranges)
creator.process_grid()


#%%

from Starfish.emulator import Emulator

emu = Emulator.from_grid("IGRINS_grid.hdf5")
print(emu)

#%%
emu.train(options=dict(maxiter=1e5))
print(emu)

from Starfish.emulator.plotting import plot_emulator

plot_emulator(emu)

#%%

emu.save("IGRINS_emu.hdf5")


#%%

from Starfish.models import SpectrumModel
model = SpectrumModel(
    "IGRINS_emu.hdf5",
    data,
    grid_params=[5000, 1.0, 0],
    Av=0,
    global_cov=dict(log_amp=38, log_ls=2),)

print(model)

#%%

model.plot();


#%%

import scipy.stats as st

priors = {
    "T": st.uniform(4000, 5500),
    "Z": st.uniform(-0.5, 0.5),
    "Av": st.halfnorm(0, 0.2),
    "global_cov:log_amp": st.norm(38, 1),
    "global_cov:log_ls": st.uniform(0, 10),
    "logg": st.uniform(0.5, 2.0),
    }

model.train(priors)

#%%

print(model)
model.plot()

#%%

model.save("HD108477_MAP.toml")

#%%

import emcee
print(emcee.__version__)

model.load("HD108477_MAP.toml")
model.freeze("global_cov")
print(model.labels)

#%%

# Set our walkers and dimensionality
nwalkers = 50
ndim = len(model.labels)

# Initialize gaussian ball for starting point of walkers
scales = {"T": 1, "Av": 0.01, "Z": 0.01}

ball = np.random.randn(nwalkers, ndim)

for i, key in enumerate(model.labels):
    ball[:, i] *= scales[key]
    ball[:, i] += model[key]

#%%
    
def log_prob(P, priors):
    model.set_param_vector(P)
    return model.log_likelihood(priors)


# Set up our backend and sampler
backend = emcee.backends.HDFBackend("HD108477_chain.hdf5")
backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_prob, args=(priors,), backend=backend)
    
#%%

max_n = 1000

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf

# Now we'll sample for up to max_n steps
for sample in sampler.sample(ball, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 10:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1
    # skip math if it's just going to yell at us
    if not np.isfinite(tau).all():
        continue
    # Check convergence
    converged = np.all(tau * 10 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        print(f"Converged at sample {sampler.iteration}")
        break
    old_tau = tau

#%%    

sampler.run_mcmc(backend.get_last_sample(), 100, progress=True);

#%%

import arviz as az
import corner

print(az.__version__, corner.__version__)

reader = emcee.backends.HDFBackend("HD108477_chain.hdf5")
full_data = az.from_emcee(reader, var_names=model.labels)

az.plot_trace(full_data);

#%%

tau = reader.get_autocorr_time(tol=0)
burnin = int(tau.max())
thin = int(0.3 * np.min(tau))
burn_samples = reader.get_chain(discard=burnin, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, thin=thin)

dd = dict(zip(model.labels, burn_samples.T))
burn_data = az.from_dict(dd)

az.plot_trace(burn_data);

#%%

print(az.summary(burn_data))

#%%

az.plot_posterior(burn_data, ["T", "Z", "logg", "Av"]);


#%%

# See https://corner.readthedocs.io/en/latest/pages/sigmas.html#a-note-about-sigmas
sigmas = ((1 - np.exp(-0.5)), (1 - np.exp(-2)))
corner.corner(
    burn_samples.reshape((-1, 3)),
    labels=model.labels,
    quantiles=(0.05, 0.16, 0.84, 0.95),
    levels=sigmas,
    show_titles=True,);

#%%

best_fit = dict(az.summary(burn_data)["mean"])
model.set_param_dict(best_fit)
print(model)

model.plot()

#%%


model.save("example_sampled.toml")

#%% 
import dill
dill.dump_session('last_env.db')
