import h5py
import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *
from shutil import copyfile
from time import gmtime, strftime

datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

    #machine can be 'BrushlessMotor', 'RoboticArm'
def preprocess_IMAD_DS(machine):

    # Initializations
    INPUT_FOLDER = f'data/IMAD-DS/{machine}'
    OUTPUT_FOLDER = f'processed/IMAD-DS_{machine}'

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # constants
    # Duration of initial data time affected by the gyroscope warm-up period
    GYROSCOPE_WARM_UP_TIME = pd.to_timedelta('35ms')
    WINDOW_SIZE_TS = pd.to_timedelta('100ms')

    # Initialize utility dictionary for preprocessing operations
    sensor_dict = {
        'imp23absu_mic': {
            'fs': 16000,
            'number_of_channel': 1
        },
        'ism330dhcx_acc': {
            'fs': 7063,  # Estimated sampling rate calculated by averaging time deltas across all files
            'number_of_channel': 3
        },
        'ism330dhcx_gyro': {
            'fs': 7063,  # Estimated sampling rate calculated by averaging time deltas across all files
            'number_of_channel': 3
        }
    }

    for sensor in sensor_dict.keys():
        sensor = sensor_dict[sensor]
        sensor['window_length'] = int(
            sensor['fs'] * WINDOW_SIZE_TS.total_seconds())
    
    normal_source_train = pd.read_csv(
        f'{INPUT_FOLDER}/train/attributes_normal_source_train.csv',
        index_col=0)
    normal_target_train = pd.read_csv(
        f'{INPUT_FOLDER}/train/attributes_normal_target_train.csv',
        index_col=0)
    normal_source_test = pd.read_csv(
        f'{INPUT_FOLDER}/test/attributes_normal_source_test.csv',
        index_col=0)
    anomaly_source_test = pd.read_csv(
        f'{INPUT_FOLDER}/test/attributes_anomaly_source_test.csv',
        index_col=0)
    normal_target_test = pd.read_csv(
        f'{INPUT_FOLDER}/test/attributes_normal_target_test.csv',
        index_col=0)
    anomaly_target_test = pd.read_csv(
        f'{INPUT_FOLDER}/test/attributes_anomaly_target_test.csv',
        index_col=0)

    Train_Metadata = pd.concat(
        [normal_source_train, normal_target_train], axis=0).reset_index(drop=True)
    Test_Metadata = pd.concat([normal_source_test,
                            anomaly_source_test,
                            normal_target_test,
                            anomaly_target_test],
                            axis=0).reset_index(drop=True)

    # create segment id column
    Train_Metadata['segment_id'] = Train_Metadata['imp23absu_mic'].apply(
        lambda x: x.replace('imp23absu_mic_', ''))
    Test_Metadata['segment_id'] = Test_Metadata['imp23absu_mic'].apply(
        lambda x: x.replace('imp23absu_mic_', ''))

    # add customized path to each filepath in the Metadata dataframes
    for sensor in sensor_dict.keys():
        Train_Metadata[sensor] = INPUT_FOLDER + '/train/' + Train_Metadata[sensor]
        Test_Metadata[sensor] = INPUT_FOLDER + '/test/' + Test_Metadata[sensor]

    # Loop through each dataset split type ('train' and 'test') with
    # corresponding metadata
    for split_type, metadata in zip(
            ['train', 'test'], [Train_Metadata, Test_Metadata]):
        # Define the save path for the HDF5 file
        save_path = '{}/{}_dataset_window_{:.3f}s.h5'.format(
            OUTPUT_FOLDER,
            split_type,
            WINDOW_SIZE_TS.total_seconds()
        )

        print(save_path)

        # Open the HDF5 file in write mode
        with h5py.File(save_path, 'w') as h5file:
            # ================================================================ INIT
            # Initialize datasets dictionary to store HDF5 datasets
            datasets = {}

            # Create datasets for each sensor defined in sensor_dict
            for sensor in sensor_dict.keys():
                window_length = sensor_dict[sensor]['window_length']
                number_of_channel = sensor_dict[sensor]['number_of_channel']

                # Create a dataset for each sensor with specified shape and
                # chunking
                datasets[sensor] = h5file.create_dataset(
                    sensor,
                    shape=(0, number_of_channel, window_length),
                    maxshape=(None, number_of_channel, window_length),
                    chunks=True
                )

            # Create additional datasets for segment ID and various labels

            # dataset containing the index of corresponding segment
            datasets['segment_id'] = h5file.create_dataset(
                'segment_id',
                shape=(0, 1),
                maxshape=(None, 1),
                chunks=True,
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            # dataset containing split labels
            datasets['split_label'] = h5file.create_dataset(
                'split_label',
                shape=(0, 1),
                maxshape=(None, 1),
                chunks=True,
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            # dataset containing anomaly labels
            datasets['anomaly_label'] = h5file.create_dataset(
                'anomaly_label',
                shape=(0, 1),
                maxshape=(None, 1),
                chunks=True,
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            # dataset containing operational domain shift labels
            datasets['domain_shift_op'] = h5file.create_dataset(
                'domain_shift_op',
                shape=(0, 1),
                maxshape=(None, 1),
                chunks=True,
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            # dataset containing environmental domain shift labels
            datasets['domain_shift_env'] = h5file.create_dataset(
                'domain_shift_env',
                shape=(0, 1),
                maxshape=(None, 1),
                chunks=True,
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            # ============================================  DATA SEGMENTATION INTO
            # Every row of the Metadata represent the i-th segment of one specific recording:
            # the same segment is recorded for all sensors, named in the same way and its path
            # is linked in the appropriate column of the dataframe

            # Iterate over all segments in the metadata
            for file_index in range(len(metadata)):
                try:
                    print(
                        f'Completed: {file_index / (len(metadata)-1)*100:.2f}%',
                        end='\r')

                    # Load and process data for each sensor
                    for sensor in sensor_dict:
                        sensor_df = pd.read_parquet(metadata[sensor][file_index])
                        sensor_df['Time'] = pd.to_datetime(
                            sensor_df['Time'], unit='s')
                        sensor_df.set_index('Time', inplace=True)
                        sensor_df.sort_index(inplace=True)

                        sensor_dict[sensor]['data_raw'] = sensor_df
                        sensor_dict[sensor]['max_ts'] = sensor_df.index[-1]
                        sensor_dict[sensor]['min_ts'] = sensor_df.index[0]

                    # Determine the time range for the segment: makes sure that
                    # there is available data for all sensors
                    max_ts_list = [sensor_dict[sensor]['max_ts']
                                for sensor in sensor_dict]
                    min_ts_list = [sensor_dict[sensor]['min_ts']
                                for sensor in sensor_dict]

                    start_timestamp = max(
                        sensor_dict['ism330dhcx_gyro']['min_ts'] +
                        GYROSCOPE_WARM_UP_TIME,
                        max(min_ts_list))
                    end_timestamp = min(max_ts_list)

                    # Extract labels for the segment
                    segment_id = metadata['segment_id'][file_index]
                    split_label = metadata['split_label'][file_index]
                    anomaly_label = metadata['anomaly_label'][file_index]
                    domain_shift_op = metadata['domain_shift_op'][file_index]
                    domain_shift_env = metadata['domain_shift_env'][file_index]

                    flag = 1
                    number_of_window = (
                        end_timestamp - start_timestamp) // WINDOW_SIZE_TS

                    # Iterate over each sensor to process the data into windows
                    for sensor in sensor_dict:
                        sensor_df = sensor_dict[sensor]['data_raw']
                        num_points_per_window = sensor_dict[sensor]['window_length']
                        num_channel = sensor_dict[sensor]['number_of_channel']

                        # Iterate over each window in the segment
                        for window_idx in range(number_of_window):
                            start = start_timestamp + window_idx * WINDOW_SIZE_TS
                            end = start + WINDOW_SIZE_TS
                            sensor_df_window = sensor_df[start:end].values

                            # Zero-pad or truncate the window to match the expected
                            # length
                            l = len(sensor_df_window)
                            if l < num_points_per_window:
                                pad_size = num_points_per_window - l
                                padding = np.zeros((pad_size, num_channel))
                                sensor_df_window = np.vstack(
                                    [sensor_df_window, padding])
                            else:
                                sensor_df_window = sensor_df_window[:num_points_per_window, :]

                            # Resize and store the windowed data in the HDF5
                            # dataset
                            current_size = datasets[sensor].shape[0]
                            datasets[sensor].resize(current_size + 1, axis=0)
                            datasets[sensor][-1] = sensor_df_window.T

                            if flag: #compute only during windowing the first sensor, since the labels are equl for every sensor
                                current_size = datasets['segment_id'].shape[0]

                                datasets['segment_id'].resize(
                                    current_size + 1, axis=0)
                                datasets['segment_id'][-1] = segment_id

                                datasets['split_label'].resize(
                                    current_size + 1, axis=0)
                                datasets['split_label'][-1] = split_label

                                datasets['anomaly_label'].resize(
                                    current_size + 1, axis=0)
                                datasets['anomaly_label'][-1] = anomaly_label

                                datasets['domain_shift_op'].resize(
                                    current_size + 1, axis=0)
                                datasets['domain_shift_op'][-1] = domain_shift_op

                                datasets['domain_shift_env'].resize(
                                    current_size + 1, axis=0)
                                datasets['domain_shift_env'][-1] = domain_shift_env

                        flag = 0

                except Exception as e:
                    print('could not read file index {}'.format(file_index), e)

        with h5py.File(save_path, 'r') as h5file:	
			# Save the data to a NumPy file
            data = h5file['imp23absu_mic'][:]
            np.save(OUTPUT_FOLDER + os.sep + split_type + '_imp23absu_mic' + '.npy', data)

            data = h5file['ism330dhcx_acc'][:]
            np.save(OUTPUT_FOLDER + os.sep + split_type + '_ism330dhcx_acc' + '.npy', data)

            data = h5file['ism330dhcx_gyro'][:]
            np.save(OUTPUT_FOLDER + os.sep + split_type + '_ism330dhcx_gyro' + '.npy', data)

            if split_type == 'test':
                data = h5file['anomaly_label'][:]
                np.save(OUTPUT_FOLDER + os.sep + 'labels_multisensor.npy', data, allow_pickle=True)


def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
	temp = np.zeros(shape)
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
	a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
	return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = min(a), max(a)
	return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def convertNumpy(df):
	x = df[df.columns[3:]].values[::10, :]
	return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_data(dataset):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	
	if dataset == 'synthetic':
		train_file = os.path.join(data_folder, dataset, 'synthetic_data_with_anomaly-s-1.csv')
		test_labels = os.path.join(data_folder, dataset, 'test_anomaly.csv')
		dat = pd.read_csv(train_file, header=None)
		split = 10000
		train = normalize(dat.values[:, :split].reshape(split, -1))
		test = normalize(dat.values[:, split:].reshape(split, -1))
		lab = pd.read_csv(test_labels, header=None)
		lab[0] -= split
		labels = np.zeros(test.shape)
		for i in range(lab.shape[0]):
			point = lab.values[i][0]
			labels[point-30:point+30, lab.values[i][1:]] = 1
		test += labels * np.random.normal(0.75, 0.1, test.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
			
	elif dataset=='IMAD-DS_RoboticArm' or dataset=='IMAD-DS_BrushlessMotor':
		
		machine = dataset.split('_')[1]
		preprocess_IMAD_DS(machine)
		
		dataset_folder = os.path.join('processed', dataset)
		
		train_multisensor = []
		for file in [f for f in os.listdir(dataset_folder) if 'train' in f if'.npy' in f if not 'label' in f]:
			train = np.load(os.path.join(dataset_folder,file))
			train, min_a, max_a = normalize3(train.T) # per window normalization
			train = train.T # transpose again to get original shape (rows= windows)
			np.save(os.path.join(dataset_folder,file), train)
			train = train.reshape(train.shape[0], -1) # concatenate simultaneous channels windows 
			train_multisensor.append(train)
		
		train_multisensor = np.concatenate(train_multisensor, axis=1) # concatenate all sensors into a single vector (1 row = all sensors channels concatenated)
		np.save(os.path.join(dataset_folder,'train_multisensor.npy'), train_multisensor)
		
		test_multisensor = []
		for file in [f for f in os.listdir(dataset_folder) if 'test' in f if'.npy' in f if not 'label' in f]:
			test = np.load(os.path.join(dataset_folder,file))
			test, min_a, max_a = normalize3(test.T) # per window normalization
			test = test.T # transpose again to get original shape (rows= windows)
			np.save(os.path.join(dataset_folder,file), test)
			test = test.reshape(test.shape[0], -1) # concatenate simultaneous channels windows 
			test_multisensor.append(test)
		
		test_multisensor = np.concatenate(test_multisensor, axis=1) # concatenate all sensors into a single vector (1 row = all sensors channels concatenated)
		np.save(os.path.join(dataset_folder,'test_multisensor.npy'), test_multisensor)

	elif dataset == 'SMD':
		dataset_folder = 'data/SMD'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
				s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
				load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
	elif dataset == 'UCR':
		dataset_folder = 'data/UCR'
		file_list = os.listdir(dataset_folder)
		for filename in file_list:
			if not filename.endswith('.txt'): continue
			vals = filename.split('.')[0].split('_')
			dnum, vals = int(vals[0]), vals[-3:]
			vals = [int(i) for i in vals]
			temp = np.genfromtxt(os.path.join(dataset_folder, filename),
								dtype=np.float64,
								delimiter=',')
			min_temp, max_temp = np.min(temp), np.max(temp)
			temp = (temp - min_temp) / (max_temp - min_temp)
			train, test = temp[:vals[0]], temp[vals[0]:]
			labels = np.zeros_like(test)
			labels[vals[1]-vals[0]:vals[2]-vals[0]] = 1
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
	elif dataset == 'NAB':
		dataset_folder = 'data/NAB'
		file_list = os.listdir(dataset_folder)
		with open(dataset_folder + '/labels.json') as f:
			labeldict = json.load(f)
		for filename in file_list:
			if not filename.endswith('.csv'): continue
			df = pd.read_csv(dataset_folder+'/'+filename)
			vals = df.values[:,1]
			labels = np.zeros_like(vals, dtype=np.float64)
			for timestamp in labeldict['realKnownCause/'+filename]:
				tstamp = timestamp.replace('.000000', '')
				index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
				labels[index-4:index+4] = 1
			min_temp, max_temp = np.min(vals), np.max(vals)
			vals = (vals - min_temp) / (max_temp - min_temp)
			train, test = vals.astype(float), vals.astype(float)
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			fn = filename.replace('.csv', '')
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
	elif dataset == 'MSDS':
		dataset_folder = 'data/MSDS'
		df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
		df_test  = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
		df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
		_, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
		train, _, _ = normalize3(df_train, min_a, max_a)
		test, _, _ = normalize3(df_test, min_a, max_a)
		labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
		labels = labels.values[::1, 1:]
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
	elif dataset == 'SWaT':
		dataset_folder = 'data/SWaT'
		file = os.path.join(dataset_folder, 'series.json')
		df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
		df_test  = pd.read_json(file, lines=True)[['val']][7000:12000]
		train, min_a, max_a = normalize2(df_train.values)
		test, _, _ = normalize2(df_test.values, min_a, max_a)
		labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset in ['SMAP', 'MSL']:
		dataset_folder = 'data/SMAP_MSL'
		file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
		values = pd.read_csv(file)
		values = values[values['spacecraft'] == dataset]
		filenames = values['chan_id'].values.tolist()
		for fn in filenames:
			train = np.load(f'{dataset_folder}/train/{fn}.npy')
			test = np.load(f'{dataset_folder}/test/{fn}.npy')
			train, min_a, max_a = normalize3(train)
			test, _, _ = normalize3(test, min_a, max_a)
			np.save(f'{folder}/{fn}_train.npy', train)
			np.save(f'{folder}/{fn}_test.npy', test)
			labels = np.zeros(test.shape)
			indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
			indices = indices.replace(']', '').replace('[', '').split(', ')
			indices = [int(i) for i in indices]
			for i in range(0, len(indices), 2):
				labels[indices[i]:indices[i+1], :] = 1
			np.save(f'{folder}/{fn}_labels.npy', labels)
	elif dataset == 'WADI':
		dataset_folder = 'data/WADI'
		ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
		train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
		test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
		train.dropna(how='all', inplace=True); test.dropna(how='all', inplace=True)
		train.fillna(0, inplace=True); test.fillna(0, inplace=True)
		test['Time'] = test['Time'].astype(str)
		test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
		labels = test.copy(deep = True)
		for i in test.columns.tolist()[3:]: labels[i] = 0
		for i in ['Start Time', 'End Time']: 
			ls[i] = ls[i].astype(str)
			ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
		for index, row in ls.iterrows():
			to_match = row['Affected'].split(', ')
			matched = []
			for i in test.columns.tolist()[3:]:
				for tm in to_match:
					if tm in i: 
						matched.append(i); break			
			st, et = str(row['Start Time']), str(row['End Time'])
			labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
		train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
		print(train.shape, test.shape, labels.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'MBA':
		dataset_folder = 'data/MBA'
		ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
		train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
		test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
		train, test = train.values[1:,1:].astype(float), test.values[1:,1:].astype(float)
		train, min_a, max_a = normalize3(train)
		test, _, _ = normalize3(test, min_a, max_a)
		ls = ls.values[:,1].astype(int)
		labels = np.zeros_like(test)
		for i in range(-20, 20):
			labels[ls + i, :] = 1
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
			
	else:
		raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
	commands = sys.argv[1:]
	load = []
	if len(commands) > 0:
		for d in commands:
			load_data(d)
	else:
		print("Usage: python preprocess.py <datasets>")
		print(f"where <datasets> is space separated list of {datasets}")