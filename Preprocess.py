import os
import os.path as op
import sys

import numpy as np
from scipy.signal import find_peaks

import mne
from mne.datasets import fetch_fsaverage


def get_FS_data():
    """Obtains the directory of the trans, src, bem and subjects_dir.

    :return: trans, src, bem, subjects_dir (all in strings)
    """
    # Use the fetch_fsaverage function from MNE to get fs data
    fs_dir = fetch_fsaverage(verbose=True)

    # Obtain the subject directory name
    subjects_dir = op.dirname(fs_dir)

    # Set the directory of the trans, src and bem
    trans = op.join(fs_dir, 'bem', 'fsaverage-trans.fif')
    src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    return trans, src, bem, subjects_dir


def load_raw_data(subjects_list, dataset='cursor'):
    """Loads the raw data of the subjects and returns the raw data list and the events list.

    :param subjects_list: The full list of subject
    :param dataset: Type of dataset ('cursor' or 'robot')
    :return: raw_data_list, events_list
    """
    # Current directory
    current_dir = os.getcwd()

    # Cursor directory
    if dataset == 'cursor':
        file_dir = op.join(current_dir, 'Data/cursor/processed_data')
    elif dataset == 'robot':
        file_dir = op.join(current_dir, 'Data/robot/processed_data')
    else: sys.exit('Dataset do not exists!')

    # Set the full subjects list
    if subjects_list is None:
        subjects_list = ['s02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12', 's13']

    # Initialize empty list
    raw_data_list = []
    events_list = []

    for i in range(len(subjects_list)):
        # Read individual subject file path
        if dataset == 'cursor':
            file_path = os.path.join(file_dir, subjects_list[i] + '_processed_cursor.set')
        elif dataset == 'robot':
            file_path = os.path.join(file_dir, subjects_list[i] + '_processed_robot.set')

        # Print out for the processing subject
        subject = file_path.split('/s')[2].split('_')[0]
        print('\nProcessing Subject ' + str(subject) + '\n')

        # Read the data for individual subject
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose='INFO')

        # Append the raw data to the list
        raw_data_list.append(raw)

        # Read the events
        events = mne.events_from_annotations(raw)[0]

        # Append the events to event_lists
        events_list.append(events)
    return raw_data_list, events_list


def obtain_data(subject_to_process, raw_data_list, events_list):
    """Set the subject to process and return the individual raw data.

    :param subject_to_process: The list of subject to process
    :param raw_data_list: Obtained from load_raw_data function
    :param events_list: Obtained from load_raw_data function
    :return: raw_event_dict (raw data and event of subjects to be processed)
    """
    # Initialize empty dictionary
    raw_event_dict = {}

    for i in range(len(subject_to_process)):

        print('\nCommence processing for subject set: ' + subject_to_process[i] + '\n')

        # Raw data of the individual subject
        raw = raw_data_list[i]
        event = events_list[i]

        # Read and set the EEG electrode locations
        montage = mne.channels.make_standard_montage('easycap-M1', head_size=0.095)

        # Set the montage
        raw.set_montage(montage)

        # Set the EEG reference
        raw.set_eeg_reference(projection=True)  # Required for inverse modeling

        # Append the raw and events for individual subject
        raw_event_dict[subject_to_process[i] + str('_raw')] = raw
        raw_event_dict[subject_to_process[i] + str('_event')] = event

        print('\nFinished processing for subject set: ' + subject_to_process[i])
    return raw_event_dict


def baseline_correction(epo):
    """Performs the baseline correction.

    :param epo: Data epoch
    :return: Data epoch (baseline corrected)
    """
    # Load the error and non error epochs
    epo_nE = epo['nonError'].get_data()
    epo_Er = epo['error'].get_data()

    # Correction for non error epochs
    trials, ch, signals = epo['nonError'].get_data().shape
    time_array = epo['nonError'].times
    time_idx_1 = np.where(time_array <= 0)
    time_idx_2 = np.where(time_array >= -0.2)

    for i in range(trials):
        for j in range(ch):
            epo_nE[i, j, :] = epo_nE[i, j, :] - np.mean(epo_nE[i, j, time_idx_2 and time_idx_1])

    # Correction for non error epochs
    trials, ch, signals = epo['error'].get_data().shape
    time_array = epo['error'].times
    time_idx_1 = np.where(time_array <= 0)
    time_idx_2 = np.where(time_array >= -0.2)

    for i in range(trials):
        for j in range(ch):
            epo_Er[i, j, :] = epo_Er[i, j, :] - np.mean(epo_Er[i, j, time_idx_2 and time_idx_1])

    # Update the epoch data
    epo['nonError'].get_data = epo_nE
    epo['error'].get_data = epo_Er
    return epo


def create_epochs(raw_event_dict, subject_to_process, tmin=None, tmax=None, nonError=None, error=None):
    """Create epochs based on the epoch time (tmin and tmax) and based on events.

    :param raw_event_dict: Obtained from obtain_data function
    :param subject_to_process: The list of subject to process
    :param tmin: Starting time of epoch
    :param tmax: End time of epoch
    :param nonError: ('S  4')
    :param error: ('S  5' or 'S  6')
    :return: data - {'epo': epo, 'epo_nE_avg': epo_nE_avg, 'epo_Er_avg':epo_Er_avg, 'cov': cov, 'contrast':contrast} (Nested dictionary)
    """
    # Epochs time
    if tmin is None:
        tmin = -0.2

    if tmax is None:
        tmax = 0.8

    # Initialize empty dictionary
    data = {}

    for i in range(len(subject_to_process)):
        print('\nCreating epochs for subject: ' + subject_to_process[i] + '\n')

        # Unpack the raw data and event from the dictionary
        raw = raw_event_dict[subject_to_process[i] + '_raw']
        event = raw_event_dict[subject_to_process[i] + '_event']

        # Set the event id
        if nonError == 'S  4':
            # nonError_id = mne.events_from_annotations(raw)[1]['S  4']  # 10
            nonError_id = mne.events_from_annotations(raw)[1]['FB_S4']  # 1

        if error == 'S  5':
            # error_id = mne.events_from_annotations(raw)[1]['S  5']  # 11
            error_id = mne.events_from_annotations(raw)[1]['FB_S5']  # 2

        elif error == 'S  6':
            # error_id = mne.events_from_annotations(raw)[1]['S  6']  # 12
            error_id = mne.events_from_annotations(raw)[1]['FB_S6']  # 12

        event_condition_id = dict(nonError=nonError_id, error=error_id)

        # Create the epochs
        epo = mne.Epochs(raw, events=event, event_id=event_condition_id, tmin=tmin, tmax=tmax)

        # Compute covariance matrix
        cov = mne.compute_covariance(epo, method='auto')

        # Perform baseline correction
        epo = baseline_correction(epo)

        # Average the epochs
        epo_nE_avg = epo['nonError'].average()
        epo_Er_avg = epo['error'].average()

        # Create averages for different events
        # evoked = [epo[k].average() for k in event_condition_id]
        evoked = [epo_nE_avg, epo_Er_avg]

        # Combine Evoked (Merge evoked data by weighted addition or subtraction)
        # Combined to a difference average with the weights -1 for non-error and +1 for error average
        contrast = mne.combine_evoked(evoked, weights=[-1, 1])

        # Appending the data for individual subject
        data[subject_to_process[i]] = {'epo': epo, 'epo_nE_avg': epo_nE_avg, 'epo_Er_avg': epo_Er_avg, 'cov': cov,
                                       'contrast': contrast}

        print('\nFinished creating epochs for subject: ' + subject_to_process[i])
    return data


def obtain_cz_cache(grand_average_cache_cursor, grand_average_cache_robot):
    """Obtains the cz_cache.

    :param grand_average_cache_cursor: The grand average across all cursor subjects of Cz channel
    :param grand_average_cache_robot:T he grand average across all robot subjects of Cz channel
    :return: cz_data_cache (Dictionary)
    """
    # Unpack the grand average cache
    epo_nE_grand_avg_cursor = grand_average_cache_cursor['epo_nE_grand_avg']
    epo_Er_grand_avg_cursor = grand_average_cache_cursor['epo_Er_grand_avg']
    contrast_grand_average_cursor = grand_average_cache_cursor['contrast_grand_average']

    epo_nE_grand_avg_robot = grand_average_cache_robot['epo_nE_grand_avg']
    epo_Er_grand_avg_robot = grand_average_cache_robot['epo_Er_grand_avg']
    contrast_grand_average_robot = grand_average_cache_robot['contrast_grand_average']

    # Obtain the Cz channel data
    non_error_cz_cursor = epo_nE_grand_avg_cursor.pick_channels(['Cz'])
    error_cz_cursor = epo_Er_grand_avg_cursor.pick_channels(['Cz'])
    contrast_cz_cursor = contrast_grand_average_cursor.pick_channels(['Cz'])

    non_error_cz_robot = epo_nE_grand_avg_robot.pick_channels(['Cz'])
    error_cz_robot = epo_Er_grand_avg_robot.pick_channels(['Cz'])
    contrast_cz_robot = contrast_grand_average_robot.pick_channels(['Cz'])

    # Set the timing
    t_min = -200
    t_max = 800

    # Obtain the cursor and robot data
    non_error_cz_cursor_data = non_error_cz_cursor.data[0]
    error_cz_cursor_data = error_cz_cursor.data[0]
    contrast_cz_cursor_data = contrast_cz_cursor.data[0]

    non_error_cz_robot_data = non_error_cz_robot.data[0]
    error_cz_robot_data = error_cz_robot.data[0]
    contrast_cz_robot_data = contrast_cz_robot.data[0]

    # Set the time vector
    time_pts = np.linspace(t_min, t_max, contrast_cz_cursor_data.shape[0])

    # Find peaks and min
    peak_idx = find_peaks(contrast_cz_cursor_data)[0]
    min_idx = np.argmin(contrast_cz_cursor_data)

    # Find the highest three peaks and min
    min_data_val = contrast_cz_cursor_data[min_idx]
    all_peak_vals = contrast_cz_cursor_data[peak_idx]
    sorted_peak_vals = np.sort(all_peak_vals)
    peak_data_val = sorted_peak_vals[::-1][:3]

    # Find the time points of the three peaks
    time_pt_1 = time_pts[contrast_cz_cursor_data == peak_data_val[0]][0]
    time_pt_2 = time_pts[contrast_cz_cursor_data == peak_data_val[1]][0]
    time_pt_3 = time_pts[contrast_cz_cursor_data == peak_data_val[2]][0]
    peak_data_time = [time_pt_1, time_pt_2, time_pt_3]
    min_data_time = time_pts[min_idx]

    # Cache the data
    cz_data_cache = {'non_error_cz_cursor_data': non_error_cz_cursor_data,
                     'error_cz_cursor_data': error_cz_cursor_data,
                     'contrast_cz_cursor_data': contrast_cz_cursor_data,
                     'non_error_cz_robot_data': non_error_cz_robot_data,
                     'error_cz_robot_data': error_cz_robot_data,
                     'contrast_cz_robot_data': contrast_cz_robot_data,
                     'peak_data_val': peak_data_val,
                     'peak_data_time': peak_data_time,
                     'min_data_val': min_data_val,
                     'min_data_time': min_data_time,
                     'time_pts': time_pts}
    return cz_data_cache
