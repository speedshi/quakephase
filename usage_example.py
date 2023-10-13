

from quakephase import quakephase
from obspy import read


stream = read("./example_data/*")
stream.plot()

stream.detrend('simple')
stream.filter('bandpass', freqmin=2, freqmax=40)
stream.plot()

output = quakephase.qkphase(stream, file_para='./parameters.yaml')

output['prob'].plot()
for ipick in output['pick']:
    print(ipick)


