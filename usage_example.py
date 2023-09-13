

from quakephase import qkp_apply
from obspy import read


stream = read("./example_data/*")
stream.plot()
output = qkp_apply.qkphase(stream, file_para='./parameters.yaml')

stream.detrend('simple')
stream.filter('bandpass', freqmin=2, freqmax=40)
stream.plot()
output = qkp_apply.qkphase(stream, file_para='./parameters.yaml')

for ipick in output.picks:
    print(ipick)


