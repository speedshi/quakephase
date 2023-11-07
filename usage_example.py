

from quakephase import quakephase
from obspy import read


stream = read("./example_data_2/*")
stream.plot()

stream.detrend('simple')
stream.filter('bandpass', freqmin=2, freqmax=40)
stream.plot()

output = quakephase.apply(stream, file_para='./parameters.yaml')

output['prob'].plot()
for ipick in output['pick']:
    print(ipick)


