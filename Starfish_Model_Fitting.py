# -*- coding: utf-8 -*-
"""
Much of this code was taken directly from the examples for fitting a single 
order spectrum at 
https://starfish.readthedocs.io/en/latest/examples/single.html

Fitting for best-fit parameter values of Teff, logg, and Av.

Created on Thu Apr 14 12:02:45 2020

@author: Christian

"""

import numpy as np
from Starfish.grid_tools import download_PHOENIX_models


#%%

# Download Pheonix models with specified ranges
ranges = [[5700, 8600], [4.0, 6.0], [-0.5,0.5]]
download_PHOENIX_models(path="PHOENIX", ranges=ranges)

#%%

# Setup a grid interface with the models
from Starfish.grid_tools import PHOENIXGridInterfaceNoAlpha
grid = PHOENIXGridInterfaceNoAlpha(path="PHOENIX")

# Setup an HDF5 interface in order to allow much quicker reading and writing 
# than compared to loading FITS files over and over again.
from Starfish.grid_tools.instruments import SPEX
from Starfish.grid_tools import HDF5Creator
creator = HDF5Creator(
    grid, "F_SPEX_grid.hdf5",instrument=SPEX(), 
    wl_range=(0.9e4, np.inf), ranges=ranges)
creator.process_grid()

#%%

# use the HDF5 Interface to consrtuct the spectral emulator
from Starfish.emulator import Emulator
emu = Emulator.from_grid("F_SPEX_grid.hdf5")
print(emu)

#%%

# train the emulator (PCA) 
emu.train(options=dict(maxiter=1e5))
print(emu) 

# check that it trained properly, the GPs should have smooth lines with small 
# errors conecting the weights
from Starfish.emulator.plotting import plot_emulator
plot_emulator(emu)

#%%

# save the emulator to pass to the SpectrumModel constructor
emu.save("F_SPEX_emu.hdf5")


#%%

# load the example spectrum data
from Starfish.spectrum import Spectrum
data = Spectrum.load("example_spec.hdf5")
data.plot()

#%%

# construct the SpectrumModel with initial parameter guesses
from Starfish.models import SpectrumModel
model = SpectrumModel(
    "F_SPEX_emu.hdf5",
    data,
    grid_params=[6800, 4.2, 0], #[Teff, logg, Z]
    Av=0,
    global_cov=dict(log_amp=38, log_ls=2),)

print(model)

#%%

# plot our data and the model we have constructed
model.plot();

#%%

# fix the logg parameter so it is not optimized 
model.freeze("logg")
print(model.labels)

#%%

# define our priors on every model parameter
import scipy.stats as st
priors = {
    "T": st.norm(6800, 100),
    "Z": st.uniform(-0.5, 0.5),
    "Av": st.halfnorm(0, 0.2),
    "global_cov:log_amp": st.norm(38, 1),
    "global_cov:log_ls": st.uniform(0, 10),
    }

# perform maximum-likelihood estimate for model parameters
model.train(priors)

#%%

# plot the new "optimized" model
print(model)
model.plot()

#%%

# save this model as a starting point for MCMC
model.save("example_MAP.toml")

#%%

# load the model and freeze the global_cov parameters
model.load("example_MAP.toml")
model.freeze("global_cov")
print(model.labels)

#%%

import emcee

# Set the number of walkers and dimensions
nwalkers = 50
ndim = len(model.labels)

# Initialize gaussian ball for starting point of walkers
scales = {"T": 1, "Av": 0.01, "Z": 0.01}
ball = np.random.randn(nwalkers, ndim)
for i, key in enumerate(model.labels):
    ball[:, i] *= scales[key]
    ball[:, i] += model[key]

#%%

# define log_prob function to pass to emcee by making use of the model object's
# builtin function
def log_prob(P, priors):
    model.set_param_vector(P)
    return model.log_likelihood(priors)


# Set up our backend and ensemble sampler
backend = emcee.backends.HDFBackend("example_chain.hdf5")
backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_prob, args=(priors,), backend=backend)
    
#%%

# define max number of iterations
max_n = 1000

# track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# useful for testing convergence
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

# gather 100 more samples to ensure clean chains
sampler.run_mcmc(backend.get_last_sample(), 100, progress=True);

#%%

import arviz as az
import corner

# plot chains
reader = emcee.backends.HDFBackend("example_chain.hdf5")
full_data = az.from_emcee(reader, var_names=model.labels)
az.plot_trace(full_data);

#%%

# remove burn-in data and thin and replot
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

# print MCMC summary
print(az.summary(burn_data))

#%%

# plot the marginalized posteriors for each parameter
az.plot_posterior(burn_data, ["T", "Z", "Av"]);


#%%

# See https://corner.readthedocs.io/en/latest/pages/sigmas.html#a-note-about-sigmas
sigmas = (0.34, 0.68)
corner.corner(
    burn_samples.reshape((-1, 3)),
    labels=model.labels,
    #quantiles=(0.05, 0.16, 0.84, 0.95),
    levels=sigmas,
    show_titles=True,);

#%%

# create best-fit model
best_fit = dict(az.summary(burn_data)["mean"])
model.set_param_dict(best_fit)
print(model)

model.plot()

#%%

# save best-fit model
model.save("example_sampled.toml")

#%%
