import os
import pickle
import hashlib
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import pypsg
import pdb

# DIRECTORY:
input_dir = r'C:/Users/janee/Documents/Astrophotonics/ETC/AWG-ETC/inputs/'
output_dir = r'C:/Users/janee/Documents/Astrophotonics/ETC/AWG-ETC/outputs/'

psg = pypsg.PSG(timeout_seconds=120)




def get_cache_key(config):
    """
    Generate a unique hash key for the PSG config parameters.
    """
    
    relevant_keys = [
        'OBJECT-NAME', 'GENERATOR-RANGE1', 'GENERATOR-RANGE2', 'GENERATOR-RESOLUTION',
        'GENERATOR-NOISETIME', 'OBJECT-SEASON', 'GENERATOR-RADUNITS'
    ]
    key_str = "_".join([str(config.get(k, '')) for k in relevant_keys])
    
    return hashlib.md5(key_str.encode()).hexdigest()

def cleanup_cache(cache_dir, max_age_days=7):
    """
    Remove cache files older than max_age_days.
    """
    if not os.path.exists(cache_dir):
        return
    now = time.time()
    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        if os.path.isfile(filepath):
            file_age_days = (now - os.path.getmtime(filepath)) / 86400
            if file_age_days > max_age_days:
                os.remove(filepath)
                print(f"Removed old cache file: {filename}")

def call_psg(flux_units, phase, Plot=True, cache_dir=None, cleanup=False, max_cache_age_days=7):
    """
    Calls PSG using pypsg package. Loads cached result if available, otherwise calls the API and caches result.
    Returns filtered wavelength/transit data.
    """
    
    if cache_dir is None:
        cache_dir = os.path.join(output_dir, 'psg_cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # optional: clean up old cache files
    if cleanup:
        cleanup_cache(cache_dir, max_age_days=max_cache_age_days)
    
    
    
    config = psg.default_config.copy()
    
    # set parameters (all the same for ETC and planetary, except phase and units)
    config['OBJECT'] = 'Exoplanet'
    config['OBJECT-GRAVITY-UNIT'] = 'kg'
    config['GEOMETRY'] = 'Observatory'
    config['GENERATOR-RANGE1'] = 1.2
    config['GENERATOR-RANGE2'] = 1.7
    config['GENERATOR-RANGEUNIT'] = 'um'
    config['GENERATOR-RESOLUTION'] = 30000
    config['GENERATOR-RESOLUTIONUNIT'] = 'RP'
    config['GENERATOR-TRANS'] = '02-01'
    config['GENERATOR-TRANS-SHOW'] = 'Y'
    config['GENERATOR-TRANS-APPLY'] = 'Y'
    config['GENERATOR-NOISEFRAMES'] = 20
    config['GENERATOR-NOISE'] = 'CCD'
    config['GEOMETRY-ALTITUDE-UNIT'] = 'pc'
    
    # user parameters
    config['OBJECT-NAME'] = 'WASP-127b'
    config['OBJECT-DIAMETER'] = 183307  # [km]
    config['OBJECT-GRAVITY'] = 3.1264467e26 # mass of object [kg]
    config['OBJECT-STAR-DISTANCE'] = 0.04840 # semi-major axis [AU]
    config['OBJECT-STAR-VELOCITY'] = 0.022 # RV amplitude [km/s]
    config['OBJECT-INCLINATION'] = 87.84
    config['OBJECT-STAR-TYPE'] = 'G'
    config['OBJECT-STAR-TEMPERATURE'] = 5828 # [K]
    config['OBJECT-STAR-RADIUS'] = 0.9228 #  [Rsun]
    config['OBJECT-SEASON'] = phase # 180: secondary transit. 90: opposition
    config['GEOMETRY-OBS-ALTITUDE'] = 159.507 # [pc]
    config['GENERATOR-RADUNITS'] = flux_units # 'Wsrm2um', 'pm', or 'Jy'
    config['GENERATOR-NOISETIME'] = 10 # exposure time per frame [s]
    
    
    # generate cache key and path
    cache_key = get_cache_key(config)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    # check if result is cached
    if os.path.exists(cache_path):
        print("Loading PSG result from cache...")
        with open(cache_path, 'rb') as f:
            data_array = pickle.load(f)
    else:
        # call PSG API
        print("Calling PSG API...")
        result = psg.run(config)
        
        
        # --------- DEBUGGING -------------
        print("First 5 lines of PSG spectrum data:")
        if isinstance(result['spectrum'], np.ndarray):
            print(result['spectrum'][:5])
        else:
            print(result['spectrum'][:500])
        if not isinstance(result['spectrum'], np.ndarray):
            print("PSG returned a non-numeric response or error message:")
            print(result['spectrum'])
            raise RuntimeError("PSG did not return numeric data. Check server status or your request parameters.")
        print('\nDuration (seconds):\n' + str(result['duration_seconds']))
        
        
        data = result['spectrum']
        wavelength = data[:, 0]
        radiance_total = data[:, 1]
        radiance_noise = data[:, 2]
        radiance_stellar = data[:, 3]
        radiance_exoplanet = data[:, 4]
        
        # determine if phase=180 or phase=90 (transit radiance column)
        if data.shape[1] == 6:
            radiance_transit = data[:, 5]
        else:
            radiance_transit = None
            
        # create data_array with all PSG data   
        if radiance_transit is not None:
            data_array = np.vstack((wavelength, radiance_total, radiance_noise, radiance_stellar, radiance_exoplanet, radiance_transit))
        else:
            data_array = np.vstack((wavelength, radiance_total, radiance_noise, radiance_stellar, radiance_exoplanet))
        data_array = data_array.T
        
        # save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(data_array, f)
     
    
    # filter top 100 OH emission regions
    filtered_data = filter_spectrum(data_array)

        
            
    # PLOT (optional)            
    if Plot:
        plt.figure(figsize=(10,6))
        plt.plot(filtered_data[:,0], filtered_data[:,1], label='Total Radiance', color='blue')
        plt.plot(filtered_data[:,0], filtered_data[:,3], label='Stellar Radiance', color='orange')
        plt.plot(filtered_data[:,0], filtered_data[:,4], label='Wasp-127b Radiance', color='green')
        plt.plot(filtered_data[:,0], filtered_data[:,2], label='Noise', color='black')
        if data_array.shape[1] > 5:
            plt.plot(filtered_data[:,0], filtered_data[:,5], label='Transit Radiance', color='red')        
        
        plt.xlabel('Wavelength [µm]')
        if flux_units=='pm':
            plt.ylabel('Flux [photons measured]')
        elif flux_units=='Wsrm2um':
            plt.ylabel('Flux [W/sr/µm/m^2]')
        else:
            plt.ylabel('Flux')
        plt.title('{} Spectrum (Phase={}, Exposure time={})'.format(config['OBJECT-NAME'], phase, config['GENERATOR-NOISETIME']))
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}plots/{config['OBJECT-NAME']}_{config['GENERATOR-RESOLUTION']}_{config['GENERATOR-NOISETIME']}_{phase}_{flux_units}.png", dpi=300)
        plt.show()
        
    return filtered_data


# return wavelength, snr
def exposure_time_calculator(t):
    # FILE NAMES:
    injection_fn = 'Scaled_efficiency.csv'     # 1000-1376 nm
    pl_smf_fn = 'W_vs_PT_1cmleadin.csv'        # 1000-1400 nm
    chip_fn = 'Converted_Transmission.csv'   # 1200-1642 nm
    chip_fsr_fn = 'SiO2_refractive_index.csv'   # 0-3 microns
    
    
    print('Calling PSG for stellar data.')
    etc_data = call_psg('Jy', 90, Plot=False)
    
    wavelength_data = etc_data[:,0]   # [microns]
    stellar_flux = etc_data[:,3] # [Jy]
    
    
    spectrum_data = (stellar_flux * (3e-5)) / ((wavelength_data*(1e4))**2)  # Jy -> [erg/s/cm^2/A]
    spectrum_data = spectrum_data/(1e-8)                               # [erg/s/cm^2/A] -> [erg/s/cm^2/cm]
    wavelength_data = wavelength_data * (1e-4)   # [micron] -> [cm]

    
    # set constants
    c = 29979245800 #[cm/s]
    h = 6.6261*(1e-27) #[erg-s]


    # ------ INJECTION ------ 
    injection_data = pd.read_csv(input_dir + injection_fn) # 77 data points
    injection_wl = np.array(injection_data['Wavelength (nm)'].tolist())* (1e-7)  # [nm] -> [cm]
    injection_eff = np.array(injection_data['Efficiency (%)'].tolist())
    linear_interp = interp1d(injection_wl, injection_eff, kind='linear', fill_value='extrapolate')
    injection_transmission = linear_interp(wavelength_data)


    # ------ PL TO SMF ------ 
    pl_smf_data = pd.read_csv(input_dir + pl_smf_fn) # 78 data points
    pl_smf_wl = np.array(pl_smf_data['Wavelength'].tolist())* (1e-7)  # [nm] -> [cm]
    pl_smf_thru = np.array(pl_smf_data['Photonic Throughput'].tolist())
    linear_interp = interp1d(pl_smf_wl, pl_smf_thru, kind='linear', fill_value='extrapolate')
    PLSMF_transmission = linear_interp(wavelength_data)


    # ------ ON CHIP ------ 
    chip_data = pd.read_csv(input_dir + chip_fn) # 77 data points
    chip_wl = np.array(chip_data['Wavelength (nm)'].tolist())* (1e-7)  # [nm] -> [cm]
    chip_eff = np.array(chip_data['Transmission (Power Ratio)'].tolist())
    linear_interp = interp1d(chip_wl, chip_eff, kind='linear', fill_value='extrapolate')
    onchip_transmission = linear_interp(wavelength_data)


    # ------ CHIP TO FREE SPACE ------ 
    chip_fsr_data = pd.read_csv(input_dir + chip_fsr_fn)
    chip_fsr_wl = np.array(chip_fsr_data['Wavelength (um)'].tolist())* (1e-4)  # [um] -> [cm]
    chip_fsr_n =  np.array(chip_fsr_data['Refractive Index'].tolist())
    n_2 = 1            # refractive index for air
    n_1 = chip_fsr_n   # refractive index for SiO2(λ)

    # use fresnel equation for normal incidence
    chip_fsr_R = ((n_1 - n_2)/(n_1 + n_2))**2
    chip_fsr_T = 1 - chip_fsr_R  # 58 data points
    linear_interp = interp1d(chip_fsr_wl, chip_fsr_T, kind='linear', fill_value='extrapolate')
    chipFSR_transmission = linear_interp(wavelength_data)


    # ------ OTHERS: ------ 
    SMFchip_transmission = 0.9
    QE_efficiency = 0.9

    #total_transmission
    total_transmission = (injection_transmission * PLSMF_transmission * SMFchip_transmission *
                        onchip_transmission * chipFSR_transmission * QE_efficiency)


    A_T = 7.854e5 # total light collecting area (D=10m) [cm^2] 
    E_ph = (h*c)/(wavelength_data) # energy per photon [ergs]
    K_0 = (total_transmission*A_T)/(E_ph) # K_0 term defined for simplicity
    resolving_power = 30000 # [dimensionless]
    delta_lambda = (wavelength_data)/resolving_power # [cm]


    # total signal
    signal = np.abs(K_0 * spectrum_data * t * delta_lambda)
            
    
    A_bg = 1.6e-3 # [arcseconds^2] - area of the sky over which the flux is summed
    n_pix = 2 # total number of pixels used in measuring flux
    readout_noise = 100/QE_efficiency
    dark_current =  15/QE_efficiency
    sky_background = (2.1) * 1e-23 * (c/(wavelength_data)**2) #[Jy per arcsecond^2] -> [cgs per arcsecond^2]


    # total noise
    noise = np.sqrt(np.abs(K_0 * (spectrum_data + (A_bg * sky_background)) * delta_lambda * t
                                + n_pix * ((dark_current * t) + readout_noise**2)))

    
    # S/N
    snr = signal/noise

    plt.plot(wavelength_data*1e7, snr)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('S/N')
    plt.grid(True)
    plt.title('SNR vs. Wavelength')
    plt.show() 
    
    
    
    return wavelength_data, snr


def hitran_line_list(hitran_filename):
    """
    Processes a .out HITRAN file. Converts wavenumber to wavelength, and creates a list with
    wavelengths, intensities, and HWHM values.

    Parameters:
    -----------
    hitran_filename (str) - path to the HITRAN file to process.

    Returns:
    --------
    line_list (array-like) - [wavelength (nm), intensity, HWHM]
    """
    
    path = input_dir + hitran_filename
    with open(path, 'r') as file:
        lines = file.readlines()
        
    # var 4: wavenumber [cm^-1]
    # var 5: intensity [cm^-1/(molecule * cm^-2)] = [cm/molecule]
    # var 7: air broadened HWHM [cm^-1/atm]

    nu, intensity, HWHM = [], [], []
    for line in lines:
        if line.strip(): 
            columns = line.split()
            nu.append(float(columns[3]))
            intensity.append(float(columns[4]))
            HWHM.append(float(columns[7]))

    # conversion from wavenumber [cm^-1] to wavelength [nm]     
    wavelength = [(1/val)*1e7 for val in nu]
    wavelength = np.array(wavelength[::-1])

    # flip orders for other variables
    intensity = intensity[::-1]
    HWHM = np.array(HWHM[::-1])

    # convert HWHM from delta nu to delta lambda
    # then convert HWHM from [cm] to [nm]
    HWHM = -wavelength**2 * HWHM 
    HWHM = HWHM * (1e-7)

    line_list = list(zip(wavelength, intensity, HWHM))
    return line_list

def replace_data_around_lines(data_array, line_centers, line_hwhms, tolerance=0.25):
    """
    Filters radiance data around specified spectral lines by replacing values near 
    the line centers with the average radiance of neighboring points.

    Parameters:
    -----------
    data_array (2D numpy array) - input data array where:
        - data_array[0] : wavelength values
        - data array[1-i] : radiance values

    line_centers (list/array-like) - wavelengths of the spectral lines to replace
    line_hwhms (list/array-like) - half-width at half-maximum (HWHM) for each excluded spectral line
    phase (int) - observation phase; if 180, transit radiance exists
    tolerance (float, optional) - extra buffer added to the exclusion range (default is 0.25)

    Returns:
    --------
    modified_data_array : 2D numpy array with modified radiance values.

    Notes:
    ------
    - Radiance values within [center - 2.5*HWHM - tolerance, center + 2.5*HWHM + tolerance]
      are replaced with the average radiance of the nearest points outside the range.
    """

    print('Replacing data around lines now.')

    # make a copy to avoid modifying original
    modified_data_array = data_array.copy()
    wavelength = data_array[0]

    for center, hwhm in zip(line_centers, line_hwhms):
        lower_bound = center - 2.5 * hwhm - tolerance
        upper_bound = center + 2.5 * hwhm + tolerance

        # define region around lines
        mask = (wavelength >= lower_bound) & (wavelength <= upper_bound)
        indices = np.where(mask)[0]

        if len(indices) > 0:  # ensure mask gives us valid indices
            start_idx = indices[0]
            end_idx = indices[-1]

            # get average radiance just outside the range
            # do this over all data_array indices
            for i in range(1, modified_data_array.shape[0]): 
                avg_radiance = None
                if start_idx > 0 and end_idx < len(wavelength) - 1:
                    avg_radiance = (modified_data_array[i, start_idx - 1] + modified_data_array[i, end_idx + 1]) / 2
                elif start_idx > 0:  # edge case near start
                    avg_radiance = modified_data_array[i, start_idx - 1]
                elif end_idx < len(wavelength) - 1:  # edge case near end
                    avg_radiance = modified_data_array[i, end_idx + 1]

                # replace all values in the range with the computed average
                if avg_radiance is not None:
                    modified_data_array[i, indices] = avg_radiance

    return modified_data_array

def filter_spectrum(data, Plot=False):
    
    """
    Processes a PSG spectrum by removing contamination from the top 100 OH emission lines.
    Optionally plots the filtered spectrum and top 100 OH lines.
    
    Notes:
    --------
    Calls function remove_data_around_lines()
    Calls function hitran_line_list()
    """
    
    print('Starting filter spectrum process.')
    
    line_list = hitran_line_list(r'OH_list.out')
    
    # sort by intensity (descending order)
    top_100 = sorted(line_list, key=lambda x: x[1], reverse=True)[:100]
    top_100_wavelengths = np.array([line[0] for line in top_100])
    top_100_hwhm = [line[2] for line in top_100]

    # modify PSG data array
    modified_data = replace_data_around_lines(data, top_100_wavelengths, top_100_hwhm)
    wavelength = modified_data[0]
    modified_radiance = modified_data[1]

    if Plot:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(wavelength*1e-3, modified_radiance, label='Spectrum', linewidth=2)
        ax.vlines(
            x=top_100_wavelengths*1e-3,
            ymin=0, ymax=5e-11,
            linewidth=1, color='red', label='top 100 OH emission lines')
        [ax.spines[s].set_visible(False) for s in ['top', 'right']]
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Spectral Flux (W/μm)')
        ax.set_title('Spectra w/OH emission lines')
        ax.legend()
        plt.grid()
        plt.show()


    print('Spectrum now filtered!')

    return modified_data


def calculate_planet_snr(exoplanet_info, snr_wavelength, snr_star, Plot=True):
    
    exoplanet_name = exoplanet_info[0]
    R = exoplanet_info[1]
    t = exoplanet_info[2]
    
    print('Calling PSG for planet data.')
    planet_data = call_psg('pm', 180, Plot=False)
    
    # ----- calculate signal ratio -----
    s_p = planet_data[:,5] * (-1)
    s_star = planet_data[:,3]
    wavelength = planet_data[:,0]
    
    
    # interpolate star SNR onto the wavelength grid of your PSG data
    interp_map = interp1d(snr_wavelength*(1e4), snr_star, kind='linear', fill_value="extrapolate")
    snr_star_interp = interp_map(wavelength)
    
    # ----- calculate N_lines -----
    line_list = hitran_line_list('top_lines.out')
    
    # sort line list by intensity (descending order)
    top_100 = sorted(line_list, key=lambda x: x[1], reverse=True)[:100]
    top_100_intensities = [line[1] for line in top_100]
    
    # normalize intensities
    max_intensity = max(top_100_intensities)
    normalized_intensity = [i / max_intensity for i in top_100_intensities]

    # N_lines = summation of normalized intensities
    N_lines = sum(normalized_intensity)


    # ----- SNR EQUATION -----
    snr_planet = (s_p/s_star) * snr_star_interp * np.sqrt(N_lines)

    
    
    if Plot:
        plt.figure(figsize=(10, 6))
        plt.plot(wavelength, snr_planet, label='Planet SNR', color='blue')
        plt.xlabel('Wavelength (microns)')
        plt.ylabel('SNR')
        plt.title('{}: Wavelength vs. Planet SNR'.format(exoplanet_name))
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}plots/{exoplanet_name}_{R}_{t}_SNR.png", dpi=300)
        plt.show()
    
    return snr_planet, wavelength








if __name__ == "__main__":
    
    exposure_time = 10 # [s]
    
    # ETC S/N calculation
    etc_wavelength, etc_snr = exposure_time_calculator(exposure_time)

    # Planet S/N calculation
    exoplanet = ['WASP-127b', 30000, exposure_time]
    planet_data = calculate_planet_snr(exoplanet, etc_wavelength, etc_snr)

   
   
   
   
   