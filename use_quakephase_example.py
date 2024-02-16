

# Load quakephase and importe the required libraries
from quakephase import quakephase
from quakephase.utilplot import waveform_pick_1sta


# %% Example 1: load seismic data stored locally and apply quakephase to obtain P and S picks and probabilities
from obspy import read

# read seismic data stored locally
stream = read("./example_data_2/*")

# plot waveforms
st = stream.copy().filter('bandpass', freqmin=5, freqmax=45)
st.plot()

# apply quakephase to get P and S picks and probabilities
output = quakephase.apply(stream, file_para='./parameters.yaml')

# print/save the pick and probability
output['prob'].plot()
print(output['pick'])
output['pick'].to_csv('pick.csv', index=False)

# visualize together with phase picks and probabilities
waveform_pick_1sta(stream=stream, pick=output['pick'], prob=output['prob'], fband=[2, 40])




# %% Example 2: load seismic data from online data center and apply quakephase to obtain P and S picks and probabilities
from obspy.clients.fdsn import RoutingClient
from obspy import UTCDateTime

# request seismic data from data center
client = RoutingClient("iris-federator")
starttime = UTCDateTime("2024-01-04 06:33:00")
endtime = UTCDateTime("2024-01-04 06:46:00")
stream = client.get_waveforms(network="CH", station="HASLI", location="*", channel="HH?", starttime=starttime, endtime=endtime)

# plot waveforms
stream.plot()

# apply quakephase to get P and S picks and probabilities
output = quakephase.apply(stream, file_para='./parameters.yaml')

# print/save the pick and probability
output['prob'].plot()
print(output['pick'])
output['pick'].to_csv('pick.csv', index=False)

# visualize together with phase picks and probabilities
waveform_pick_1sta(stream=stream, pick=output['pick'], prob=output['prob'], fband=[2, 20])



