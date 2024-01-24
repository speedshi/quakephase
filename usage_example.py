

from quakephase import quakephase
from obspy import read
from quakephase.utilplot import waveform_pick_1sta


stream = read("./example_data_2/*")
stream.plot()


output = quakephase.apply(stream, file_para='./parameters.yaml')

output['prob'].plot()
print(output['pick'])
output['pick'].to_csv('pick.csv', index=False)


waveform_pick_1sta(stream=stream, pick=output['pick'], prob=output['prob'], fband=[2, 40])




