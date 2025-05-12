import pypsg


psg = pypsg.PSG(timeout_seconds=60)
config = psg.default_config
config['OBJECT-NAME'] = 'Earth'
config['OBJECT-STAR-RADIUS'] = 1.2
result = psg.run(config)


# The reply header as a string
print('\nPSG reply header:\n' + result['header'])

# The generated spectrum as a Numpy array
print('\nSpectrum\n' + str(result['spectrum']))

# The time in seconds that was consumed
print('\nDuration (seconds):\n' + str(result['duration_seconds']))



