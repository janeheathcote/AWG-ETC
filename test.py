import pypsg
import numpy as np



psg = pypsg.PSG(timeout_seconds=60)
config = psg.default_config


""" def call_psg(flux_units, phase, Plot=True):

    print('about to run call_psg function:')

    # list of things that would be set
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

    # config['GENERATOR-NOISE1'] = read noise
    # config['GENERATOR-NOISE2'] = dark current
    # config['GENERATOR-NOISEOEFF'] = throughput (wavelength dependent; see format)
    # config['GENERATOR-NOISEOEMIS'] = emissivity
    # config['GENERATOR-NOISEOTEMP'] = temp of telescope/optics [K]


    # list of things that would be changed
    config['OBJECT-NAME'] = 'WASP-127b'
    config['OBJECT-DIAMETER'] = 183307  # [km]
    config['OBJECT-GRAVITY'] = 3.1264467e26 # mass of object [kg]
    config['OBJECT-STAR-DISTANCE'] = 0.04840 # semi-major axis [AU]
    config['OBJECT-STAR-VELOCITY'] = 0.022 # RV amplitude [km/s] (not on archive)
    config['OBJECT-INCLINATION'] = 87.84
    config['OBJECT-STAR-TYPE'] = 'G'
    config['OBJECT-STAR-TEMPERATURE'] = 5828 # [K]
    config['OBJECT-STAR-RADIUS'] = 0.9228 #  [Rsun]
    config['OBJECT-SEASON'] = phase # 180: secondary transit. 90: opposition
    config['GEOMETRY-OBS-ALTITUDE'] = 159.507 # [pc] distance between observer and planet
    config['GENERATOR-RADUNITS'] = flux_units # 'Wsrm2um': flux units. 'pm': photons measured units
    config['GENERATOR-NOISETIME'] = 10 # exposure time per frame [s]


    print('about to run result line:')

    result = psg.run(config)


    # the reply header
    print('\nPSG reply header:\n' + result['header'])

    # the time in seconds
    print('\nDuration (seconds):\n' + str(result['duration_seconds']))



    data = result['spectrum']
    wavelength = data[:, 0]         # [microns]
    radiance_total = data[:, 1]     # total radiance [whichever PSG units were chosen]
    radiance_noise = data[:, 2]     # noise
    radiance_stellar = data[:, 3]   # stellar radiance
    radiance_exoplanet = data[:, 4] # exoplanet radiance

    # determine if phase=180 or phase=90 (transit radiance column)
    if data.shape[1] == 6:
        radiance_transit = data[:, 5]  # transit radiance
    else:
        radiance_transit = None


    # create data_array with all PSG data
    if radiance_transit is not None:
        data_array = np.vstack((wavelength, radiance_total, radiance_noise, radiance_stellar, radiance_exoplanet, radiance_transit))
    else:
        data_array = np.vstack((wavelength, radiance_total, radiance_noise, radiance_stellar, radiance_exoplanet))
    data_array = data_array.T  
    
    return data_array


# personal sanity check - visually inspect spectra by plotting
test = call_psg('pm', 180) """





result = psg.run(config)

# The reply header as a string
print('\nPSG reply header:\n' + result['header'])

# The generated spectrum as a Numpy array
print('\nSpectrum\n' + str(result['spectrum']))

# The time in seconds that was consumed
print('\nDuration (seconds):\n' + str(result['duration_seconds']))
