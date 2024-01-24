
from quakephase import quakephase
from quakephase.drumplot import drum_plot_one_channel_with_ml_picks
from obspy import read
from obspy import Stream

# load 1 minute data
# yaml channel list, time


stream = read("./data/*")
# stream.plot(method='full')

# Choose one sensor trace

istream = stream.select(station="F0803")
# istream.plot()

# Change the channel name from 'JJD' to 'JJZ'
sttime = istream[0].stats.starttime+1
istream_t = istream.slice(starttime=sttime, endtime=sttime+1)[0]
istream_t.stats.channel = 'JJZ'
# istream_t.plot()

# Fill into three-component format, empty traces for JJ1 & JJ2

istream_1 = istream_t.copy()
istream_1.stats.channel = 'JJ1'
istream_1.data[:] = 0

istream_2 = istream_t.copy()
istream_2.stats.channel = 'JJ2'
istream_2.data[:] = 0

istream_3c = Stream()
istream_3c.append(istream_t)
istream_3c.append(istream_1)
istream_3c.append(istream_2)
istream_3c.plot()

# ML model running
output = quakephase.apply(istream_3c, file_para='parameters.yaml')

output['prob'].plot(method='full')
for ipick in output['pick']:
    print(ipick)
    print(ipick.__dict__)

fig = drum_plot_one_channel_with_ml_picks(main_name='JJZ',stbeg=istream_3c[0].stats.starttime, stend=istream_3c[0].stats.starttime+1, stream=istream_3c, prediction_output=output,
                                          parameters_file='parameters.yaml', time_interval=0.1, pixel_per_second=2000, vert_scale=60,
                                          preprocessing=True, taper_rate=0.01, low_threshold=1e3, high_threshold=5e3, filter_corners=4,
                                          mark_length_factor=0.2, line_wid=0.8, num_ticks=6, dpi=1000, save=False)
fig.show()