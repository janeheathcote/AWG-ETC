# ETC_pipeline.py

Summary:
This script is a comprehensive pipeline for simulating and analyzing the performance of an AWG (Arrayed Waveguide Grating) spectrograph, particularly for exoplanet transit observations. It includes an Exposure Time Calculator (ETC) and additional tools for end-to-end calculation of planetary S/N.

Key Features:

    Exposure Time Calculator (ETC): Calculates the required exposure time and signal-to-noise ratio (S/N) for a given
    observation scenario.

    Remote PSG Integration: Uses the [pypsg package](https://gitlab.com/frontierdevelopmentlab/astrobiology/pypsg) to remotely
    call NASA's Planetary Spectrum Generator (PSG) for generating stellar and transit spectra. Results are cached for efficiency.

    Planetary S/N Calculation: Computes planetary signal-to-noise using a dedicated equation.

    Spectral Filtering: Removes contamination from the top 100 OH emission lines using HITRAN line lists.

    Plotting: Generates plots of spectra and S/N as a function of wavelength.

Intended Use:
Use this pipeline for full end-to-end simulations, including automated remote spectral generation and S/N analysis for exoplanet spectroscopy projects


# ETC.py

Summary:
This script is a standalone version of the Exposure Time Calculator (ETC) for an AWG spectrograph.

Key Features:

    ETC Functionality: Focuses solely on the exposure time and S/N calculations for an AWG spectrograph.

    Manual Input of Spectra: Instead of calling PSG remotely, users must upload a .txt file containing the input spectral data.

Intended Use:
Use this script if you only need the ETC calculations and already have your input spectra, or if you want to avoid dependencies on remote services.
