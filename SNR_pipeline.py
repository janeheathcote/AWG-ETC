import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pdb

# DIRECTORY:
input_dir = r'C:/Users/janee/Documents/Astrophotonics/ETC/inputs/'
output_dir = r'C:/Users/janee/Documents/Astrophotonics/ETC/outputs/'


def read_txt(filename, Plot=True):
    """
    Reads a PSG spectrum text file, extracts wavelength and radiance data, and optionally plots the spectrum.

    Parameters:
    -----------
    filename (str) - name of PSG txt file. The filename is expected to follow the format: 
        '<exoplanet_name>_<R>_<t>_<phase>_<unit>.txt', where:
        - <exoplanet_name> is the name of the exoplanet.
        - <R> is the spectral resolution.
        - <t> is the exposure time.
        - <phase> is the observational phase (e.g., 90 or 180).
        - <unit> specifies whether the data is in "flux" or "photons" units.

        If phase=180, the file contains a transit radiance column.
        If phase=90, the transit radiance column is not present.
        
    Plot (bool, optional) - whether to plot the radiance spectrum (default is True).

    Returns:
    --------
    data_array (array-like) - contains wavelength, total radiance, stellar radiance,
    exoplanet radiance, and optional transit radiance (if phase=180).
    """
    input_dir = r'C:/Users/janee/Documents/Astrophotonics/ETC/inputs/'
    data = []
    
    filename_parts = filename.split('_')
    exoplanet_name = filename_parts[0]
    R = filename_parts[1]
    t = filename_parts[2]
    phase = filename_parts[3]
    unit = filename_parts[4].split('.')[0]
    
    with open(input_dir+filename, 'r') as file:
        for line in file:
            if line.startswith("#") or line.strip()=="":
                continue
            # each txt file line -> list of strings
            row = list(map(float, line.split()))
            data.append(row)

    data = np.array(data)
    wavelength = data[:,0]          # [microns]
    radiance_total = data[:,1]      # total radiance [whichever PSG units were chosen]
    radiance_noise = data[:,2]      # noise
    radiance_stellar = data[:,3]    # stellar radiance 
    radiance_exoplanet = data[:,4]  # wasp-127b radiance
    
    # if phase=180, define radiance_transit
    try:
        phase = int(phase)
    except ValueError:
        raise ValueError(f"Invalid phase value in filename: {phase}")
    
    if phase == 180:
        radiance_transit = data[:, 5]  # transit radiance
    else:
        radiance_transit = None
    
    
    if Plot==True:
        plt.figure(figsize=(10,6))
        plt.plot(wavelength, radiance_total, label='Total Radiance', color='blue')
        plt.plot(wavelength, radiance_stellar, label='Stellar Radiance', color='orange')
        plt.plot(wavelength, radiance_exoplanet, label='Wasp-127b Radiance', color='green')
        
        if radiance_transit is not None:
            plt.plot(wavelength, radiance_transit, label='Transit Radiance', color='red')
        
        plt.plot(wavelength, radiance_noise, label='Noise', color='black')
                
        plt.xlabel('Wavelength [µm]')
        if unit=='photons':
            plt.ylabel('Flux [photons measured]')
        elif unit=='flux':
            plt.ylabel('Flux [W/sr/µm/m^2]')
        else:
            plt.ylabel('Flux')
        plt.title('{} Spectrum (Phase={}, R={}, Exposure time={})'.format(exoplanet_name, phase, R, t))
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}plots/{exoplanet_name}_{R}_{t}_{phase}_{unit}.png", dpi=300)
        plt.show()
    
    # create data_array with all PSG data
    if radiance_transit is not None:
        data_array = np.vstack((wavelength, radiance_total, radiance_noise, radiance_stellar, radiance_exoplanet, radiance_transit))
    else:
        data_array = np.vstack((wavelength, radiance_total, radiance_noise, radiance_stellar, radiance_exoplanet))
        
    return data_array

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

    # make a copy to avoid modifying original
    modified_data_array = data_array.copy()
    wavelength = data_array[0]*1000

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

def filter_spectrum(psg_txt_filename, Plot=False):
    """
    Processes a PSG spectrum by removing contamination from the top 100 OH emission lines.
    Optionally plots the filtered spectrum and top 100 OH lines.

    Parameters:
    -----------
    psg_txt_filename (str) - path to the PSG spectrum file to process.
    Plot (bool, optional) - whether to plot the filtered spectrum (default is False).

    Returns:
    --------
    filtered_wavelength (array-like)
    filtered_radiance (array-like)
    
    Notes:
    --------
    Calls function read_txt()
    Calls function remove_data_around_lines()
    Calls function hitran_line_list()
    """
    
    line_list = hitran_line_list(r'OH_list.out')
    
    # sort by intensity (descending order)
    top_100 = sorted(line_list, key=lambda x: x[1], reverse=True)[:100]
    top_100_wavelengths = np.array([line[0] for line in top_100])
    top_100_hwhm = [line[2] for line in top_100]

    # read in PSG spectra
    data = read_txt(psg_txt_filename, Plot=False)
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

    return modified_data

def calculate_snr_star(star_data):
    """
    Reads a PSG txt file (with "photons measured" units) and calculates the signal/noise.
    
    Parameters:
    -----------
    star_data (2D numpy array) - input data array where:
        - data_array[0] : wavelength values
        - data_array[1] : total radiance
        - data_array[2] : noise
        - data_array[3] : stellar radiance
        - data_array[4] : exoplanet radiance

    Returns:
    --------
    snr_wavelength (array-like) - the wavelength array
    snr (array-like) - the SNR per spectral resolution element
    """
    
    signal = star_data[1]    # total radiance [photons measured]
    noise = star_data[2]     # noise [photons measured]
    
    snr = signal/noise
    return snr

def calculate_snr_planet(filename, transit_data, star_data, Plot=True):
    """
    Calculates the signal-to-noise ratio (SNR) of a planet.

    Parameters:
    -----------
    filename (string): name of PSG txt file. The filename is expected to follow the format 
        '<exoplanet_name>_<R>_<t>_[...].txt', where:
        - <exoplanet_name> is the name of the exoplanet.
        - <R> is the spectral resolution.
        - <t> is the exposure time.
        
    star_data (2D numpy array) - input data array with "photons measured" units.
        - data_array[0] : wavelength values
        - data_array[1] : total radiance
        - data_array[2] : noise
        - data_array[3] : stellar radiance
        - data_array[4] : exoplanet radiance
        
    transit_data (2D numpy array) - input data array with same format as star_data,
    except with an additional column
        - data_array[5] : transit radiance
    
    Plot (bool): option to plot wavelength vs. planetary SNR. Default is True.

    Returns:
    --------
    snr_planet (array): Signal-to-noise ratio of the planet, per spectral
    resolution element.
    
    Notes:
    --------
    S/N equation from Boldt-Christmas et al. 2024 (Eq 6).
    Calls function read_txt()
    Calls function calculate_snr_star()
    Calls function hitran_line_list()
    """
    
    filename_parts = filename.split('_')
    exoplanet_name = filename_parts[0]
    R = filename_parts[1]
    t = filename_parts[2]
    
    
    # calculate signal ratio:
    s_p = transit_data[5] * (-1)
    s_star = transit_data[3]
    

    # calculate SNR of star:
    snr_star = calculate_snr_star(star_data)
    wavelength = star_data[0]
    

    # calculate N_lines:
    
    # process HITRAN line list
    line_list = hitran_line_list('top_lines.out')
    
    # sort line list by intensity (descending order)
    top_100 = sorted(line_list, key=lambda x: x[1], reverse=True)[:100]
    top_100_intensities = [line[1] for line in top_100]
    
    # normalize intensities
    max_intensity = max(top_100_intensities)
    normalized_intensity = [i / max_intensity for i in top_100_intensities]

    # N_lines = summation of normalized intensities
    N_lines = sum(normalized_intensity)


    # SNR EQUATION:
    snr_planet = (s_p/s_star) * snr_star * np.sqrt(N_lines)
    
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
    
    return snr_planet

# filenames
photons_180 = 'WASP127b_30k_10s_180_photons.txt'
photons_90 = 'WASP127b_30k_10s_90_photons.txt'
flux_90 = 'WASP127b_30k_10s_90_flux.txt'

# visually inspect the spectra
data = read_txt(photons_180)

# get data arrays + "OH line removal"
photons_transit_data = filter_spectrum(photons_180,Plot=True)
photons_star_data = filter_spectrum(photons_90)

calculate_snr_planet(photons_180, photons_transit_data, photons_star_data)


