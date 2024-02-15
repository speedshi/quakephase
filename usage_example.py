
# Load quakephase and importe the required libraries
from quakephase import quakephase
from obspy import read
from quakephase.utilplot import waveform_pick_1sta

# read seismic data and plot
stream = read("./example_data_2/*")
st = stream.copy().filter('bandpass', freqmin=5, freqmax=45)
st.plot()

# apply quakephase to pick P and S wave
output = quakephase.apply(stream, file_para='./parameters.yaml')

# print the pick and probability
output['prob'].plot()
print(output['pick'])
output['pick'].to_csv('pick.csv', index=False)

# visualize together with phase picks and probabilities
waveform_pick_1sta(stream=stream, pick=output['pick'], prob=output['prob'], fband=[2, 40])




