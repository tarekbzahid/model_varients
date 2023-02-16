import numpy as np
import pandas as pd
import h5py

df = pd.read_csv('speed2019modified.csv')
print(df.values)


#hf = h5py.File('hdf5data.h5', 'w')
#hf.create_dataset('lim_flow_data', data=df)

df.to_hdf('./speed2019modified.h5', 'df')