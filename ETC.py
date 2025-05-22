import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd


# set directory
input_dir = r'C:/Users/janee/Documents/Astrophotonics/ETC/AWG-ETC/inputs/'
output_dir = r'C:/Users/janee/Documents/Astrophotonics/ETC/AWG-ETC/outputs/'

# FILE NAMES:
injection_fn = 'Scaled_efficiency.csv'     # 1000-1376 nm
pl_smf_fn = 'W_vs_PT_1cmleadin.csv'        # 1000-1400 nm
chip_fn = 'Converted_Transmission.csv'   # 1200-1642 nm
chip_fsr_fn = 'SiO2_refractive_index.csv'   # 0-3 microns
name = r'WASP127b_30k_10s_180_Jy'

# read in PSG file:
def read_txt(filename):
    """
    Reads a PSG spectrum text file, extracts wavelength and radiance data, and optionally plots the spectrum.

    Parameters:
    -----------
    filename  - name of PSG txt file. the filename is expected to follow the format: 
              '<exoplanet_name>_<R>_<t>_<phase>.txt', where R, t, and phase represent 
               resolution, exposure time, and observational phase respectively.
        
    Returns:
    --------
    wavelength (array-like) - wavelength values [µm].
    radiance_total (array-like) - total radiance values.
    """
    data = []
    
    filename_parts = filename.split('_')
    exoplanet_name = filename_parts[0]
    #R = filename_parts[1]
    #t = filename_parts[2]
    #phase = filename_parts[3].split('.')[0]
    
    with open(input_dir+filename, 'r') as file:
        for line in file:
            if line.startswith("#") or line.strip()=="":
                continue
            # each txt file line -> list of strings
            row = list(map(float, line.split()))
            data.append(row)

    data = np.array(data)
    wavelength = data[:,0]          # [microns]
    radiance_total = data[:,1]      # total radiance [ppm]
    #radiance_stellar = data[:,3]    # stellar radiance [ppm]
    #radiance_exoplanet = data[:,4]  # exoplanet radiance [ppm]
    #radiance_transit = data[:,5]    # transit radiance [ppm]
        
    return wavelength, radiance_total


# read in data. wavelength_data: [um], spectrum_data: [Jy]
wavelength_data, spectrum_data = read_txt(name+'.txt')

spectrum_data = (spectrum_data * (3e-5)) / ((wavelength_data*(1e4))**2)  # Jy -> [erg/s/cm^2/A]
spectrum_data = spectrum_data/(1e-8)                                     # [erg/s/cm^2/A] -> [erg/s/cm^2/cm]
wavelength_data = wavelength_data * (1e-4)                               # [micron] -> [cm]


# set constants
c = 29979245800 #[cm/s]
h = 6.6261*(1e-27) #[erg-s]
exposure_time = 10  # [seconds]
resolving_power = 30000 # [dimensionless]
delta_lambda = (wavelength_data)/resolving_power # [cm]
teff = 5828 # [K], PSG WASP127b specific value


# ------ INJECTION ------ 
injection_data = pd.read_csv(input_dir + injection_fn) # 77 data points
injection_wl = np.array(injection_data['Wavelength (nm)'].tolist())* (1e-7)  # [nm] -> [cm]
injection_eff = np.array(injection_data['Efficiency (%)'].tolist())
# interpolation
# x = np.linspace(1001, 1375, 10345)
x = np.linspace(1205*(1e-7), 1375*(1e-7), 10345)
linear_interp = interp1d(injection_wl, injection_eff, kind='linear')
injection_transmission = linear_interp(x)


# ------ PL TO SMF ------ 
pl_smf_data = pd.read_csv(input_dir + pl_smf_fn) # 78 data points
pl_smf_wl = np.array(pl_smf_data['Wavelength'].tolist())* (1e-7)  # [nm] -> [cm]
pl_smf_thru = np.array(pl_smf_data['Photonic Throughput'].tolist())
# interpolation
# x = np.linspace(1000, 1398, 10345)
linear_interp = interp1d(pl_smf_wl, pl_smf_thru, kind='linear')
PLSMF_transmission = linear_interp(x)


# ------ ON CHIP ------ 
chip_data = pd.read_csv(input_dir + chip_fn) # 77 data points
chip_wl = np.array(chip_data['Wavelength (nm)'].tolist())* (1e-7)  # [nm] -> [cm]
chip_eff = np.array(chip_data['Transmission (Power Ratio)'].tolist())
# interpolation
# x = np.linspace(1202, 1642, 10345)
linear_interp = interp1d(chip_wl, chip_eff, kind='linear')
onchip_transmission = linear_interp(x)


# ------ CHIP TO FREE SPACE ------ 
chip_fsr_data = pd.read_csv(input_dir + chip_fsr_fn)
chip_fsr_wl = np.array(chip_fsr_data['Wavelength (um)'].tolist())* (1e-4)  # [um] -> [cm]
chip_fsr_n =  np.array(chip_fsr_data['Refractive Index'].tolist())

n_2 = 1            # refractive index for air
n_1 = chip_fsr_n   # refractive index for SiO2(λ)

# use fresnel equation for normal incidence
chip_fsr_R = ((n_1 - n_2)/(n_1 + n_2))**2
chip_fsr_T = 1 - chip_fsr_R  # 58 data points

# interpolation
# x = np.linspace(1000, 2000, 10345)
linear_interp = interp1d(chip_fsr_wl, chip_fsr_T, kind='linear')
chipFSR_transmission = linear_interp(x)


# ------ OTHERS (still constant for now): ------ 
SMFchip_transmission = 0.9
QE_efficiency = 0.9




# cut everything down to same wavelength range (1205-1375 nm)
wavelength = x
delta_lambda_array = wavelength/resolving_power
spectrum = np.interp(wavelength, wavelength_data, spectrum_data)

#total_transmission accounts for light transmisison through the system, dimensionless
total_transmission = (injection_transmission * PLSMF_transmission * SMFchip_transmission *
                      onchip_transmission * chipFSR_transmission * QE_efficiency)


""" plt.plot(x*1e7, total_transmission)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
plt.grid(True)
plt.title('Total Transmission (no cross-dispersion, no sky)')
plt.show() """

A_T = 7.854e5 # total light collecting area (D=10m) [cm^2] 
E_ph = (h*c)/(wavelength) # energy per photon [ergs]
K_0 = (total_transmission*A_T)/(E_ph) # K_0 term defined for simplicity


# total signal
signal = np.abs(K_0 * spectrum * exposure_time * delta_lambda_array) 


""" plt.plot(x*1e7, signal)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Signal')
plt.grid(True)
plt.title('Signal vs. Wavelength')
plt.show() """


#constants relevant to noise calculation
A_bg = 1.6e-3 # [arcseconds^2] - area of the sky over which the flux is summed
n_pix = 2 # total number of pixels used in measuring flux
readout_noise = 100/QE_efficiency
dark_current =  15/QE_efficiency     # QE changes from num(e-) -> num(photons)
#sky_background = (4.9e-3) * 1e-23 * (c/(x)**2) #[Jy per arcsecond^2] -> [cgs per arcsecond^2]
sky_background = (2.1) * 1e-23 * (c/(x)**2) #[Jy per arcsecond^2] -> [cgs per arcsecond^2]


noise = np.sqrt(np.abs(K_0 * (spectrum + (A_bg * sky_background)) * delta_lambda_array * exposure_time
                             + n_pix * ((dark_current * exposure_time) + readout_noise**2)))


 
 
""" plt.plot(x*1e7, noise)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Noise')
plt.grid(True)
plt.title('Noise vs. Wavelength')
plt.show()  """
        

snr = signal/noise

plt.plot(x*1e7, snr)
plt.xlabel('Wavelength (nm)')
plt.ylabel('S/N')
plt.grid(True)
plt.title('SNR vs. Wavelength')
plt.show() 

# SAVE SNR
np.savetxt(output_dir+name+'_snr.csv', np.column_stack((x*1e-4, snr)), delimiter=',', header='wavelength_um,snr', comments='')
