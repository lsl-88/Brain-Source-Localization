import os
import os.path as op

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.stats import (spatio_temporal_cluster_1samp_test, summarize_clusters_stc)

from nilearn.plotting import plot_stat_map
from nilearn.image import index_img


def plot_cz_data(cz_data_cache, save):
    """Plots the cz channel data.

    :param cz_data_cache: Cz cache data obtained from obtain_cz_cache function
    :param save: Specify to save the image
    :return: -
    """
    # Unpack the cz_cache data
    non_error_cz_cursor_data = cz_data_cache['non_error_cz_cursor_data']
    error_cz_cursor_data = cz_data_cache['error_cz_cursor_data']
    contrast_cz_cursor_data = cz_data_cache['contrast_cz_cursor_data']

    non_error_cz_robot_data = cz_data_cache['non_error_cz_robot_data']
    error_cz_robot_data = cz_data_cache['error_cz_robot_data']
    contrast_cz_robot_data = cz_data_cache['contrast_cz_robot_data']

    peak_data_val = cz_data_cache['peak_data_val']
    peak_data_time = cz_data_cache['peak_data_time']
    min_data_val = cz_data_cache['min_data_val']
    min_data_time = cz_data_cache['min_data_time']
    time_pts = cz_data_cache['time_pts']

    # Obtain current directory
    current_dir = os.getcwd()

    save_directory = current_dir + '/Saved_Images/' + 'grand_average'

    if not os.path.exists(save_directory):
        if not os.path.exists(current_dir + '/Saved_Images'):
            os.mkdir(current_dir + '/Saved_Images')
        # Make subject directory
        os.mkdir(save_directory)

    # Set the conversion factor
    conv_fac = 10 ** 6  # voltage to microvoltage

    # Plot the figure for cursor data
    plt.figure(figsize=(14, 8))
    plt.grid()
    plt.plot(time_pts, contrast_cz_cursor_data * conv_fac, label='Contrast', color='black', linestyle='--', linewidth=3)
    plt.plot(time_pts, non_error_cz_cursor_data * conv_fac, label='Non-Error', color='blue', linewidth=3)
    plt.plot(time_pts, error_cz_cursor_data * conv_fac, label='Error', color='red', linewidth=3)

    for i in range(len(peak_data_val)):
        plt.plot(peak_data_time[i], peak_data_val[i] * conv_fac, 'x', markeredgewidth=3, markersize=20)

    plt.plot(min_data_time, min_data_val * conv_fac, 'x', markeredgewidth=3, markersize=20)
    plt.xlabel('Time (ms)', fontsize=20)
    plt.ylabel('Amplitude (μV)', fontsize=20)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlim(-200, 800)
    plt.ylim(-3, 5)
    plt.title('Cz Channel (Cursor)', fontsize=24)
    plt.legend(loc='upper right', fontsize=20)

    if save is True:
        os.chdir(save_directory)
        plt.savefig('Cz_cursor_data_plot')
        os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')

    plt.show()

    # Plot the figure for robot data
    plt.figure(figsize=(14, 8))
    plt.grid()
    plt.plot(time_pts, contrast_cz_robot_data * conv_fac, label='Contrast', color='black', linestyle='--', linewidth=3)
    plt.plot(time_pts, non_error_cz_robot_data * conv_fac, label='Non-Error', color='blue', linewidth=3)
    plt.plot(time_pts, error_cz_robot_data * conv_fac, label='Error', color='red', linewidth=3)

    for i in range(len(peak_data_time)):
        time_index_peak = np.where(time_pts == peak_data_time[i])[0][0]
        plt.plot(peak_data_time[i], contrast_cz_robot_data[time_index_peak] * conv_fac, 'x', markeredgewidth=3,
                 markersize=20)

    time_index_min = np.where(time_pts == min_data_time)[0][0]
    plt.plot(min_data_time, contrast_cz_robot_data[time_index_min] * conv_fac, 'x', markeredgewidth=3, markersize=20)
    plt.xlabel('Time (ms)', fontsize=20)
    plt.ylabel('Amplitude (μV)', fontsize=20)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlim(-200, 800)
    plt.ylim(-3, 5)
    plt.title('Cz Channel (Robot)', fontsize=24)
    plt.legend(loc='upper right', fontsize=20)

    if save is True:
        os.chdir(save_directory)
        plt.savefig('Cz_robot_data_plot')
        os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')

    plt.show()
    pass


def grand_avg_topomap(grand_average_cache_cursor, grand_average_cache_robot, cz_data_cache, date_type, save):
    """Plot the topomap of the grand average.

    :param grand_average_cache_cursor: Obtained from compute_grand_average function
    :param grand_average_cache_robot: Obtained from compute_grand_average function
    :param cz_data_cache: Cz cache data obtained from obtain_cz_cache function
    :param date_type: (eg: 'Error')
    :param save: Specify to save the image
    :return: -
    """
    print('\nGenerating topomap for grand average' + '\n')

    # Obtain current directory
    current_dir = os.getcwd()

    save_directory = current_dir + '/Saved_Images/' + 'grand_average'

    if not os.path.exists(save_directory):
        if not os.path.exists(current_dir + '/Saved_Images'):
            os.mkdir(current_dir + '/Saved_Images')
        # Make subject directory
        os.mkdir(save_directory)

    # Unpack the cz_cache and append to time list
    time_list = []
    peak_data_time = cz_data_cache['peak_data_time']
    min_data_time = cz_data_cache['min_data_time']
    time_list.extend(peak_data_time)
    time_list.append(min_data_time)
    time_list = np.sort(time_list)
    new_time_list = [i / 1000 for i in time_list]

    # Unpack the grand_average_cache
    if date_type == 'contrast':
        grand_avg_cursor = grand_average_cache_cursor['contrast_grand_average']
        grand_avg_robot = grand_average_cache_robot['contrast_grand_average']
    elif date_type == 'non_error':
        grand_avg_cursor = grand_average_cache_cursor['epo_Er_grand_avg']
        grand_avg_robot = grand_average_cache_robot['epo_Er_grand_avg']
    elif date_type == 'error':
        grand_avg_cursor = grand_average_cache_cursor['epo_nE_grand_avg']
        grand_avg_robot = grand_average_cache_robot['epo_nE_grand_avg']

    # Plot 2D topography of evoked responses
    if date_type == 'contrast':
        image_cursor = grand_avg_cursor.plot_topomap(times=new_time_list, show_names=True, size=2,
                                                     title='Contrast (Cursor)', time_unit='ms')
        image_robot = grand_avg_robot.plot_topomap(times=new_time_list, show_names=True, size=2,
                                                   title='Contrast (Robot)', time_unit='ms')

    elif date_type == 'non_error':
        image_cursor = grand_avg_cursor.plot_topomap(times=new_time_list, show_names=True, size=2,
                                                     title='Non Error (Cursor)', time_unit='ms')
        image_robot = grand_avg_robot.plot_topomap(times=new_time_list, show_names=True, size=2,
                                                   title='Non Error (Robot)', time_unit='ms')

    elif date_type == 'error':
        image_cursor = grand_avg_cursor.plot_topomap(times=new_time_list, show_names=True, size=2,
                                                     title='Error(Cursor)', time_unit='ms')
        image_robot = grand_avg_robot.plot_topomap(times=new_time_list, show_names=True, size=2, title='Error (Robot)',
                                                   time_unit='ms')

    if save is True:
        os.chdir(save_directory)
        image_cursor.savefig('grand_average_' + str(date_type) + '_cursor_topoplot.png')
        image_robot.savefig('grand_average_' + str(date_type) + '_robot_topoplot.png')
        os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')
    pass


def plot_time_montage(grand_average_cache_cursor, grand_average_cache_robot, grand_average_parameters, trans, src, bem, subjects_dir):
    """Plots and saves the time montage view of all of the ErrP times of interest.

    :param grand_average_cache_cursor: Obtained from compute_grand_average function
    :param grand_average_cache_robot: Obtained from compute_grand_average function
    :param grand_average_parameters: The parameters for the plot
    :param trans: Model obtained from get_FS_data function
    :param src: Model obtained from get_FS_data function
    :param bem: Model obtained from get_FS_data function
    :param subjects_dir: Directory to the brain model
    :return: -
    """
    # Unpack the grand average parameters
    data_type = grand_average_parameters['data_type']
    method = grand_average_parameters['method']
    vector = grand_average_parameters['vector']
    cz_data_cache = grand_average_parameters['cz_data_cache']
    selected_view = grand_average_parameters['selected_view']
    hemisphere = grand_average_parameters['hemisphere']
    save_time_montage = grand_average_parameters['save_time_montage']

    # Unpack the cz_cache and append to time list
    time_list = []
    peak_data_time = cz_data_cache['peak_data_time']
    min_data_time = cz_data_cache['min_data_time']
    time_list.extend(peak_data_time)
    time_list.append(min_data_time)
    time_list = np.sort(time_list)
    new_time_list = [i / 1000 for i in time_list]

    # Unpack the grand_average_cache
    if data_type == 'contrast':
        grand_avg_cursor = grand_average_cache_cursor['contrast_grand_average']
        grand_avg_robot = grand_average_cache_robot['contrast_grand_average']
    elif data_type == 'non_error':
        grand_avg_cursor = grand_average_cache_cursor['epo_Er_grand_avg']
        grand_avg_robot = grand_average_cache_robot['epo_Er_grand_avg']
    elif data_type == 'error':
        grand_avg_cursor = grand_average_cache_cursor['epo_nE_grand_avg']
        grand_avg_robot = grand_average_cache_robot['epo_nE_grand_avg']

    cov_cursor = grand_average_cache_cursor['cov']
    cov_robot = grand_average_cache_robot['cov']

    # Obtain current directory
    current_dir = os.getcwd()
    save_directory_cursor = op.join(current_dir, 'Saved_Images/grand_average/cursor/time_montage')
    save_directory_robot = op.join(current_dir, 'Saved_Images/grand_average/robot/time_montage')

    if not os.path.exists(save_directory_cursor) or not os.path.exists(save_directory_robot):
        if not os.path.exists(current_dir + '/Saved_Images/grand_average/cursor') or not os.path.exists(
                current_dir + '/Saved_Images/grand_average/robot'):
            if not os.path.exists(current_dir + '/Saved_Images/grand_average'):
                if not os.path.exists(current_dir + '/Saved_Images'):
                    os.mkdir(current_dir + '/Saved_Images')
                os.mkdir(current_dir + '/Saved_Images/grand_average')
            os.mkdir(current_dir + '/Saved_Images/grand_average/cursor')
            os.mkdir(current_dir + '/Saved_Images/grand_average/robot')
        # Make save directory
        os.mkdir(save_directory_cursor)
        os.mkdir(save_directory_robot)

    # Perform Forward Modelling
    fwd_cursor = mne.make_forward_solution(grand_avg_cursor.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0,
                                           n_jobs=1)
    fwd_robot = mne.make_forward_solution(grand_avg_robot.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0,
                                          n_jobs=1)

    # Assemble inverse operator
    info_cursor = grand_avg_cursor.info
    info_robot = grand_avg_robot.info

    inv_cursor = make_inverse_operator(info_cursor, fwd_cursor, cov_cursor, loose=0.2, depth=0.8)
    inv_robot = make_inverse_operator(info_robot, fwd_robot, cov_robot, loose=0.2, depth=0.8)

    # Apply inverse operator to evoked data
    if method is None:
        method = 'dSPM'  # 'dSPM' 'MNE' 'sLORETA' 'eLORETA'

    # Perform inverse modelling
    stc_cursor = apply_inverse(grand_avg_cursor, inv_cursor, lambda2=1. / 9., method=method)
    stc_robot = apply_inverse(grand_avg_robot, inv_robot, lambda2=1. / 9., method=method)

    stc_cursor_vec = apply_inverse(grand_avg_cursor, inv_cursor, lambda2=1. / 9., method=method, pick_ori='vector')
    stc_robot_vec = apply_inverse(grand_avg_robot, inv_robot, lambda2=1. / 9., method=method, pick_ori='vector')

    # Initialize empty list
    stc_vertno_cursor_max_list = []
    stc_time_cursor_max_list = []
    stc_vertno_robot_max_list = []
    stc_time_robot_max_list = []
    delta = 0.002

    # Obtain the peak data and time
    for i in range(len(new_time_list)):
        # Set the new tmin and tmax
        new_tmin = new_time_list[i] - delta
        new_tmax = new_time_list[i] + delta

        # Get peak in the specified time range
        stc_vertno_cursor_max, stc_time_cursor_max = stc_cursor.get_peak(hemi=hemisphere, tmin=new_tmin, tmax=new_tmax)
        stc_vertno_cursor_max_list.append(stc_vertno_cursor_max)
        stc_time_cursor_max_list.append(stc_time_cursor_max)

        stc_vertno_robot_max, stc_time_robot_max = stc_robot.get_peak(hemi=hemisphere, tmin=new_tmin, tmax=new_tmax)
        stc_vertno_robot_max_list.append(stc_vertno_robot_max)
        stc_time_robot_max_list.append(stc_time_robot_max)

    # Set the surfer_kwargs
    surfer_kwargs = dict(hemi=hemisphere, subjects_dir=subjects_dir,
                         views=selected_view, time_unit='ms',
                         size=(800, 800), smoothing_steps=5)

    # Set the number of rows and columns
    ncol = len(new_time_list)

    # Set the figure size
    fig = plt.figure(figsize=(48, 12))

    # Plot and save the subplots
    for i in range(len(new_time_list)):

        plt.subplot(1, ncol, i + 1)

        if vector is False:

            brain_cursor = stc_cursor.plot(**surfer_kwargs)
            brain_cursor.set_time(time_list[i])
            brain_cursor.add_foci(stc_vertno_cursor_max_list[i], coords_as_verts=True, hemi=hemisphere, color='blue',
                                  scale_factor=0.6, alpha=0.5)
            screenshot_cursor = brain_cursor.screenshot_single()
            plt.imshow(screenshot_cursor)
            fig.suptitle(
                '[' + hemisphere.upper() + '] [Cursor] Non-Vector Solution (' + data_type.upper() + ') (' + selected_view.upper() + ' VIEW) ' + '[' + method + ']',
                fontsize=54)

            if save_time_montage is True:
                os.chdir(save_directory_cursor)
                plt.savefig(
                    'non_vector_' + data_type + '_' + selected_view + '_' + method + '_' + hemisphere + '_grand_average_cursor_time_montage')
                os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')

        else:

            brain_cursor = stc_cursor_vec.plot(**surfer_kwargs)
            brain_cursor.set_time(time_list[i])
            brain_cursor.add_foci(stc_vertno_cursor_max_list[i], coords_as_verts=True, hemi=hemisphere, color='blue',
                                  scale_factor=0.6, alpha=0.5)
            single_screenshot = brain_cursor.screenshot_single()
            plt.imshow(single_screenshot)
            fig.suptitle(
                '[' + hemisphere.upper() + '] [Cursor] Vector Solution (' + data_type.upper() + ') (' + selected_view.upper() + ' VIEW) ' + '[' + method + ']',
                fontsize=54)

            if save_time_montage is True:
                os.chdir(save_directory_cursor)
                plt.savefig(
                    'vector_' + data_type + '_' + selected_view + '_' + method + '_' + hemisphere + '_grand_average_cursor_time_montage')
                os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')

    # Set the figure size
    fig = plt.figure(figsize=(48, 12))

    # Plot and save the subplots
    for i in range(len(new_time_list)):

        plt.subplot(1, ncol, i + 1)

        if vector is False:

            brain_robot = stc_robot.plot(**surfer_kwargs)
            brain_robot.set_time(time_list[i])
            brain_robot.add_foci(stc_vertno_robot_max_list[i], coords_as_verts=True, hemi=hemisphere, color='blue',
                                 scale_factor=0.6, alpha=0.5)
            screenshot_robot = brain_robot.screenshot_single()
            plt.imshow(screenshot_robot)
            fig.suptitle(
                '[' + hemisphere.upper() + '] [Robot] Non-Vector Solution (' + data_type.upper() + ') (' + selected_view.upper() + ' VIEW) ' + '[' + method + ']',
                fontsize=54)

            if save_time_montage is True:
                os.chdir(save_directory_robot)
                plt.savefig(
                    'non_vector_' + data_type + '_' + selected_view + '_' + method + '_' + hemisphere + '_grand_average_robot_time_montage')
                os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')
        else:

            brain_robot = stc_robot_vec.plot(**surfer_kwargs)
            brain_robot.set_time(time_list[i])
            brain_robot.add_foci(stc_vertno_robot_max_list[i], coords_as_verts=True, hemi=hemisphere, color='blue',
                                 scale_factor=0.6, alpha=0.5)
            screenshot_robot = brain_robot.screenshot_single()
            plt.imshow(screenshot_robot)
            fig.suptitle(
                '[' + hemisphere.upper() + '] [Robot] Vector Solution (' + data_type.upper() + ') (' + selected_view.upper() + ' VIEW) ' + '[' + method + ']',
                fontsize=54)

            if save_time_montage is True:
                os.chdir(save_directory_robot)
                plt.savefig(
                    'vector_' + data_type + '_' + selected_view + '_' + method + '_' + hemisphere + '_grand_average_robot_time_montage')
                os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')
    pass


def save_single_image(grand_average_cache_cursor, grand_average_cache_robot, grand_average_parameters, trans, src, bem, subjects_dir):
    """saves the single view of all of the ErrP times of interest.

    :param grand_average_cache_cursor: Obtained from compute_grand_average function
    :param grand_average_cache_robot: Obtained from compute_grand_average function
    :param grand_average_parameters: The parameters for the plot
    :param trans: Model obtained from get_FS_data function
    :param src: Model obtained from get_FS_data function
    :param bem: Model obtained from get_FS_data function
    :param subjects_dir: Directory to the brain model
    :return: -
    """
    # Unpack the grand average parameters
    data_type = grand_average_parameters['data_type']
    method = grand_average_parameters['method']
    vector = grand_average_parameters['vector']
    cz_data_cache = grand_average_parameters['cz_data_cache']
    selected_view = grand_average_parameters['selected_view']
    hemisphere = grand_average_parameters['hemisphere']
    save = grand_average_parameters['save']

    # Unpack the cz_cache and append to time list
    time_list = []
    peak_data_time = cz_data_cache['peak_data_time']
    min_data_time = cz_data_cache['min_data_time']
    time_list.extend(peak_data_time)
    time_list.append(min_data_time)
    time_list = np.sort(time_list)
    new_time_list = [i / 1000 for i in time_list]

    # Unpack the grand_average_cache
    if data_type == 'contrast':
        grand_avg_cursor = grand_average_cache_cursor['contrast_grand_average']
        grand_avg_robot = grand_average_cache_robot['contrast_grand_average']
    elif data_type == 'non_error':
        grand_avg_cursor = grand_average_cache_cursor['epo_Er_grand_avg']
        grand_avg_robot = grand_average_cache_robot['epo_Er_grand_avg']
    elif data_type == 'error':
        grand_avg_cursor = grand_average_cache_cursor['epo_nE_grand_avg']
        grand_avg_robot = grand_average_cache_robot['epo_nE_grand_avg']

    cov_cursor = grand_average_cache_cursor['cov']
    cov_robot = grand_average_cache_robot['cov']

    # Obtain current directory
    current_dir = os.getcwd()
    save_directory_cursor = op.join(current_dir, 'Saved_Images/grand_average/cursor/single_view_montage')
    save_directory_robot = op.join(current_dir, 'Saved_Images/grand_average/robot/single_view_montage')

    if not os.path.exists(save_directory_cursor) or not os.path.exists(save_directory_robot):
        if not os.path.exists(current_dir + '/Saved_Images/grand_average/cursor') or not os.path.exists(
                current_dir + '/Saved_Images/grand_average/robot'):
            if not os.path.exists(current_dir + '/Saved_Images/grand_average'):
                if not os.path.exists(current_dir + '/Saved_Images'):
                    os.mkdir(current_dir + '/Saved_Images')
                os.mkdir(current_dir + '/Saved_Images/grand_average')
            os.mkdir(current_dir + '/Saved_Images/grand_average/cursor')
            os.mkdir(current_dir + '/Saved_Images/grand_average/robot')
        # Make save directory
        os.mkdir(save_directory_cursor)
        os.mkdir(save_directory_robot)

    # Perform Forward Modelling
    fwd_cursor = mne.make_forward_solution(grand_avg_cursor.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0,
                                           n_jobs=1)
    fwd_robot = mne.make_forward_solution(grand_avg_robot.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0,
                                          n_jobs=1)

    # Assemble inverse operator
    info_cursor = grand_avg_cursor.info
    info_robot = grand_avg_robot.info

    inv_cursor = make_inverse_operator(info_cursor, fwd_cursor, cov_cursor, loose=0.2, depth=0.8)
    inv_robot = make_inverse_operator(info_robot, fwd_robot, cov_robot, loose=0.2, depth=0.8)

    # Apply inverse operator to evoked data
    if method is None:
        method = 'dSPM'  # 'dSPM' 'MNE' 'sLORETA' 'eLORETA'

    # Perform inverse modelling
    stc_cursor = apply_inverse(grand_avg_cursor, inv_cursor, lambda2=1. / 9., method=method)
    stc_robot = apply_inverse(grand_avg_robot, inv_robot, lambda2=1. / 9., method=method)

    stc_cursor_vec = apply_inverse(grand_avg_cursor, inv_cursor, lambda2=1. / 9., method=method, pick_ori='vector')
    stc_robot_vec = apply_inverse(grand_avg_robot, inv_robot, lambda2=1. / 9., method=method, pick_ori='vector')

    # Initialize empty list
    stc_vertno_cursor_max_list = []
    stc_time_cursor_max_list = []
    stc_vertno_robot_max_list = []
    stc_time_robot_max_list = []
    delta = 0.002

    # Obtain the peak data and time
    for i in range(len(new_time_list)):
        # Set the new tmin and tmax
        new_tmin = new_time_list[i] - delta
        new_tmax = new_time_list[i] + delta

        # Get peak in the specified time range
        stc_vertno_cursor_max, stc_time_cursor_max = stc_cursor.get_peak(hemi=hemisphere, tmin=new_tmin, tmax=new_tmax)
        stc_vertno_cursor_max_list.append(stc_vertno_cursor_max)
        stc_time_cursor_max_list.append(stc_time_cursor_max)

        stc_vertno_robot_max, stc_time_robot_max = stc_robot.get_peak(hemi=hemisphere, tmin=new_tmin, tmax=new_tmax)
        stc_vertno_robot_max_list.append(stc_vertno_robot_max)
        stc_time_robot_max_list.append(stc_time_robot_max)

    # Set the surfer_kwargs
    surfer_kwargs = dict(hemi=hemisphere, subjects_dir=subjects_dir,
                         views=selected_view, time_unit='ms',
                         size=(800, 800), smoothing_steps=5)

    # Plot and save the subplots
    for i in range(len(new_time_list)):

        if vector is False:

            brain_cursor = stc_cursor.plot(**surfer_kwargs)
            brain_cursor.set_time(time_list[i])
            brain_cursor.add_foci(stc_vertno_cursor_max_list[i], coords_as_verts=True, hemi=hemisphere, color='blue',
                                  scale_factor=0.6, alpha=0.5)
            brain_cursor.add_text(0.1, 0.9,
                                  '[' + hemisphere.upper() + '] [Cursor] Non-Vector Solution (' + data_type.upper() + ') (' + selected_view.upper() + ' VIEW) ' + '[' + method + ']',
                                  'title', font_size=84)

            if save is True:
                os.chdir(save_directory_cursor)
                brain_cursor.save_imageset(
                    'non_vector_' + data_type + '_' + selected_view + '_' + method + '_' + hemisphere + '_' + str(
                        time_list[i]) + '_ms_grand_average_cursor.png', views=[selected_view])
                os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')

        else:

            brain_cursor = stc_cursor_vec.plot(**surfer_kwargs)
            brain_cursor.set_time(time_list[i])
            brain_cursor.add_foci(stc_vertno_cursor_max_list[i], coords_as_verts=True, hemi=hemisphere, color='blue',
                                  scale_factor=0.6, alpha=0.5)
            brain_cursor.add_text(0.1, 0.9,
                                  '[' + hemisphere.upper() + '] [Cursor] Vector Solution (' + data_type.upper() + ') (' + selected_view.upper() + ' VIEW) ' + '[' + method + ']',
                                  'title', font_size=84)

            if save is True:
                os.chdir(save_directory_cursor)
                brain_cursor.save_imageset(
                    'vector_' + data_type + '_' + selected_view + '_' + method + '_' + hemisphere + '_' + str(
                        time_list[i]) + '_ms_grand_average_cursor.png', views=[selected_view])
                os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')

    # Change back to original directory
    os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')

    for i in range(len(new_time_list)):

        if vector is False:

            brain_robot = stc_robot.plot(**surfer_kwargs)
            brain_robot.set_time(time_list[i])
            brain_robot.add_foci(stc_vertno_robot_max_list[i], coords_as_verts=True, hemi=hemisphere, color='blue',
                                 scale_factor=0.6, alpha=0.5)
            brain_robot.add_text(0.1, 0.9,
                                 '[' + hemisphere.upper() + '] [Robot] Non-Vector Solution (' + data_type.upper() + ') (' + selected_view.upper() + ' VIEW) ' + '[' + method + ']',
                                 'title', font_size=84)

            if save is True:
                os.chdir(save_directory_robot)
                brain_robot.save_imageset(
                    'non_vector_' + data_type + '_' + selected_view + '_' + method + '_' + hemisphere + '_' + str(
                        time_list[i]) + '_ms_grand_average_robot.png', views=[selected_view])
                os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')

        else:

            brain_robot = stc_robot_vec.plot(**surfer_kwargs)
            brain_robot.set_time(time_list[i])
            brain_robot.add_foci(stc_vertno_robot_max_list[i], coords_as_verts=True, hemi=hemisphere, color='blue',
                                 scale_factor=0.6, alpha=0.5)
            brain_robot.add_text(0.1, 0.9,
                                 '[' + hemisphere.upper() + '] [Robot] Vector Solution (' + data_type.upper() + ') (' + selected_view.upper() + ' VIEW) ' + '[' + method + ']',
                                 'title', font_size=84)

            if save is True:
                os.chdir(save_directory_robot)
                brain_robot.save_imageset(
                    'non_vector_' + data_type + '_' + selected_view + '_' + method + '_' + hemisphere + '_' + str(
                        time_list[i]) + '_ms_grand_average_robot.png', views=[selected_view])
                os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')

        # Change back to original directory
        os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')
    pass


def save_view_montage(grand_average_cache_cursor, grand_average_cache_robot, grand_average_parameters, trans, src, bem,
                      subjects_dir):
    """Saves the view montage of the ErrP times of interest.

    :param grand_average_cache_cursor: Obtained from compute_grand_average function
    :param grand_average_cache_robot: Obtained from compute_grand_average function
    :param grand_average_parameters: The parameters for the plot
    :param trans: Model obtained from get_FS_data function
    :param src: Model obtained from get_FS_data function
    :param bem: Model obtained from get_FS_data function
    :param subjects_dir: Directory to the brain model
    :return: -
    """
    # Unpack the grand average parameters
    data_type = grand_average_parameters['data_type']
    method = grand_average_parameters['method']
    vector = grand_average_parameters['vector']
    cz_data_cache = grand_average_parameters['cz_data_cache']
    hemisphere = grand_average_parameters['hemisphere']
    save_view_montage = grand_average_parameters['save_view_montage']

    # Unpack the cz_cache and append to time list
    time_list = []
    peak_data_time = cz_data_cache['peak_data_time']
    min_data_time = cz_data_cache['min_data_time']
    time_list.extend(peak_data_time)
    time_list.append(min_data_time)
    time_list = np.sort(time_list)
    new_time_list = [i / 1000 for i in time_list]

    # Unpack the grand_average_cache
    if data_type == 'contrast':
        grand_avg_cursor = grand_average_cache_cursor['contrast_grand_average']
        grand_avg_robot = grand_average_cache_robot['contrast_grand_average']
    elif data_type == 'non_error':
        grand_avg_cursor = grand_average_cache_cursor['epo_Er_grand_avg']
        grand_avg_robot = grand_average_cache_robot['epo_Er_grand_avg']
    elif data_type == 'error':
        grand_avg_cursor = grand_average_cache_cursor['epo_nE_grand_avg']
        grand_avg_robot = grand_average_cache_robot['epo_nE_grand_avg']

    cov_cursor = grand_average_cache_cursor['cov']
    cov_robot = grand_average_cache_robot['cov']

    # Obtain current directory
    current_dir = os.getcwd()
    save_directory_cursor = op.join(current_dir, 'Saved_Images/grand_average/cursor/view_montage')
    save_directory_robot = op.join(current_dir, 'Saved_Images/grand_average/robot/view_montage')

    if not os.path.exists(save_directory_cursor) or not os.path.exists(save_directory_robot):
        if not os.path.exists(current_dir + '/Saved_Images/grand_average/cursor') or not os.path.exists(
                current_dir + '/Saved_Images/grand_average/robot'):
            if not os.path.exists(current_dir + '/Saved_Images/grand_average'):
                if not os.path.exists(current_dir + '/Saved_Images'):
                    os.mkdir(current_dir + '/Saved_Images')
                os.mkdir(current_dir + '/Saved_Images/grand_average')
            os.mkdir(current_dir + '/Saved_Images/grand_average/cursor')
            os.mkdir(current_dir + '/Saved_Images/grand_average/robot')
        # Make save directory
        os.mkdir(save_directory_cursor)
        os.mkdir(save_directory_robot)

    # Perform Forward Modelling
    fwd_cursor = mne.make_forward_solution(grand_avg_cursor.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0,
                                           n_jobs=1)
    fwd_robot = mne.make_forward_solution(grand_avg_robot.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0,
                                          n_jobs=1)

    # Assemble inverse operator
    info_cursor = grand_avg_cursor.info
    info_robot = grand_avg_robot.info

    inv_cursor = make_inverse_operator(info_cursor, fwd_cursor, cov_cursor, loose=0.2, depth=0.8)
    inv_robot = make_inverse_operator(info_robot, fwd_robot, cov_robot, loose=0.2, depth=0.8)

    # Apply inverse operator to evoked data
    if method is None:
        method = 'dSPM'  # 'dSPM' 'MNE' 'sLORETA' 'eLORETA'

    # Perform inverse modelling
    stc_cursor = apply_inverse(grand_avg_cursor, inv_cursor, lambda2=1. / 9., method=method)
    stc_robot = apply_inverse(grand_avg_robot, inv_robot, lambda2=1. / 9., method=method)

    stc_cursor_vec = apply_inverse(grand_avg_cursor, inv_cursor, lambda2=1. / 9., method=method, pick_ori='vector')
    stc_robot_vec = apply_inverse(grand_avg_robot, inv_robot, lambda2=1. / 9., method=method, pick_ori='vector')

    # Initialize empty list
    stc_vertno_cursor_max_list = []
    stc_time_cursor_max_list = []
    stc_vertno_robot_max_list = []
    stc_time_robot_max_list = []
    delta = 0.002

    # Obtain the peak data and time
    for i in range(len(new_time_list)):
        # Set the new tmin and tmax
        new_tmin = new_time_list[i] - delta
        new_tmax = new_time_list[i] + delta

        # Get peak in the specified time range
        stc_vertno_cursor_max, stc_time_cursor_max = stc_cursor.get_peak(hemi=hemisphere, tmin=new_tmin, tmax=new_tmax)
        stc_vertno_cursor_max_list.append(stc_vertno_cursor_max)
        stc_time_cursor_max_list.append(stc_time_cursor_max)

        stc_vertno_robot_max, stc_time_robot_max = stc_robot.get_peak(hemi=hemisphere, tmin=new_tmin, tmax=new_tmax)
        stc_vertno_robot_max_list.append(stc_vertno_robot_max)
        stc_time_robot_max_list.append(stc_time_robot_max)

    # Set the order
    order = [['lat', 'med'], ['ros', 'cau'], ['fro', 'par']]

    # Set the surfer_kwargs
    surfer_kwargs = dict(hemi=hemisphere, colorbar=False,
                         subjects_dir=subjects_dir, time_unit='ms',
                         size=(800, 800), smoothing_steps=5)

    if vector is False:

        for i in range(len(time_list)):

            brain_cursor = stc_cursor.plot(**surfer_kwargs)
            brain_cursor.add_foci(stc_vertno_cursor_max_list[i], coords_as_verts=True, hemi=hemisphere, color='blue',
                                  scale_factor=0.6, alpha=0.5)
            brain_cursor.set_time(time_list[i])
            brain_cursor.add_text(0.1, 0.9, '[' + str((
                                                          hemisphere.upper())) + '] ' + ' [Cursor] Non-Vector Solution' + ' (' + data_type.upper() + ') ' + ' [' + method + ']',
                                  'title', font_size=32)

            if save_view_montage is True:
                os.chdir(save_directory_cursor)
                brain_cursor.save_montage(
                    'non_vector' + '_' + method + '_' + data_type + '_' + hemisphere.upper() + '_' + str(
                        time_list[i]) + '_ms_cursor_view_montage.png', order=order)
                os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')

            # Change back to original directory
            os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')
    else:

        for i in range(len(time_list)):

            brain_cursor = stc_cursor_vec.plot(**surfer_kwargs)
            brain_cursor.add_foci(stc_vertno_cursor_max_list[i], coords_as_verts=True, hemi=hemisphere, color='blue',
                                  scale_factor=0.6, alpha=0.5)
            brain_cursor.set_time(time_list[i])
            brain_cursor.add_text(0.1, 0.9, '[' + str((
                                                          hemisphere.upper())) + '] ' + ' [Cursor] Vector Solution' + ' (' + data_type.upper() + ')' + ' [' + method + ']',
                                  'title', font_size=32)

            if save_view_montage is True:
                os.chdir(save_directory_cursor)
                brain_cursor.save_montage(
                    'vector' + '_' + method + '_' + data_type + '_' + hemisphere.upper() + '_' + str(
                        time_list[i]) + '_ms_cursor_view_montage.png', order=order)
                os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')

            # Change back to original directory
            os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')

    if vector is False:

        for i in range(len(time_list)):

            brain_robot = stc_robot.plot(**surfer_kwargs)
            brain_robot.add_foci(stc_vertno_robot_max_list[i], coords_as_verts=True, hemi=hemisphere, color='blue',
                                 scale_factor=0.6, alpha=0.5)
            brain_robot.set_time(time_list[i])
            brain_robot.add_text(0.1, 0.9, '[' + str((
                                                         hemisphere.upper())) + '] ' + ' [Robot] Non-Vector Solution' + ' (' + data_type.upper() + ') ' + ' [' + method + ']',
                                 'title', font_size=32)

            if save_view_montage is True:
                os.chdir(save_directory_robot)
                brain_robot.save_montage(
                    'non_vector' + '_' + method + '_' + data_type + '_' + hemisphere.upper() + '_' + str(
                        time_list[i]) + '_ms_robot_view_montage.png', order=order)
                os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')

            # Change back to original directory
            os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')
    else:

        for i in range(len(time_list)):

            brain_robot = stc_robot_vec.plot(**surfer_kwargs)
            brain_robot.add_foci(stc_vertno_robot_max_list[i], coords_as_verts=True, hemi=hemisphere, color='blue',
                                 scale_factor=0.6, alpha=0.5)
            brain_robot.set_time(time_list[i])
            brain_robot.add_text(0.1, 0.9, '[' + str((
                                                         hemisphere.upper())) + '] ' + ' [Cursor] Vector Solution' + ' (' + data_type.upper() + ')' + ' [' + method + ']',
                                 'title', font_size=32)

            if save_view_montage is True:
                os.chdir(save_directory_robot)
                brain_robot.save_montage(
                    'vector' + '_' + method + '_' + data_type + '_' + hemisphere.upper() + '_' + str(
                        time_list[i]) + '_ms_robot_view_montage.png', order=order)
                os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')

            # Change back to original directory
            os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')
    pass


def plot_vol_source_space(grand_average_cache_cursor, grand_average_cache_robot, cz_data_cache, modelling_parameters,
                          trans, bem, subjects_dir):
    """Plots the volume source space.

    :param grand_average_cache_cursor: Obtained from compute_grand_average function
    :param grand_average_cache_robot: Obtained from compute_grand_average function
    :param cz_data_cache: Obtained from the obtain_cz_cache function
    :param modelling_parameters: The parameters for the modelling
    :param trans: Model obtained from get_FS_data function
    :param bem: Model obtained from get_FS_data function
    :param subjects_dir: Directory to the brain model
    :return: -
    """
    # Obtain current directory
    current_dir = os.getcwd()
    save_directory = current_dir + '/Saved_Images/' + 'grand_average'

    if not os.path.exists(save_directory):
        if not os.path.exists(current_dir + '/Saved_Images'):
            os.mkdir(current_dir + '/Saved_Images')
        # Make save directory
        os.mkdir(save_directory)

    # Unpack grand average cache
    evoked_cursor = grand_average_cache_cursor['contrast_grand_average']
    cov_cursor = grand_average_cache_cursor['cov']

    evoked_robot = grand_average_cache_robot['contrast_grand_average']
    cov_robot = grand_average_cache_robot['cov']

    # Unpack the modelling parameters
    pos = modelling_parameters['pos']
    threshold = modelling_parameters['threshold']
    method = modelling_parameters['method']
    save = modelling_parameters['save']

    # Unpack the cz_cache and append to time list
    time_list = []
    peak_data_time = cz_data_cache['peak_data_time']
    min_data_time = cz_data_cache['min_data_time']
    time_list.extend(peak_data_time)
    time_list.append(min_data_time)
    time_list = np.sort(time_list)
    time_list_floor = np.floor(time_list).astype(int)
    time_list_ceil = np.ceil(time_list).astype(int)

    # Setup the volume source space
    src = mne.setup_volume_source_space('fsaverage', pos, bem=bem, subjects_dir=subjects_dir, verbose=True)

    # Perform Forward Modelling
    fwd_cursor = mne.make_forward_solution(evoked_cursor.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0,
                                           n_jobs=1)
    fwd_robot = mne.make_forward_solution(evoked_robot.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0,
                                          n_jobs=1)

    # Assemble inverse operator
    info_cursor = evoked_cursor.info
    info_robot = evoked_robot.info
    inv_cursor = make_inverse_operator(info_cursor, fwd_cursor, cov_cursor, loose=1,
                                       depth=0.8)  # Set loose=1 for volumetric, discrete, or mixed source spaces
    inv_robot = make_inverse_operator(info_robot, fwd_robot, cov_robot, loose=1,
                                      depth=0.8)  # Set loose=1 for volumetric, discrete, or mixed source spaces

    # Compute inverse solution
    stc_cursor = apply_inverse(evoked_cursor, inv_cursor, lambda2=1. / 9., method=method)
    stc_robot = apply_inverse(evoked_robot, inv_robot, lambda2=1. / 9., method=method)

    # Export result as a 4D nifti object
    img_cursor = stc_cursor.as_volume(src, mri_resolution=None)  # set True for full MRI resolution
    img_robot = stc_robot.as_volume(src, mri_resolution=None)  # set True for full MRI resolution

    # Set the directory to MRI T1 images
    t1_fname = '/home/sailam/mne_data/MNE-fsaverage-data/fsaverage/mri/T1.mgz'

    # Set the data time list from stc
    data_time_list = stc_cursor.times * 1000
    data_time_list = np.floor(data_time_list).astype(int)

    # Initialize an empty time index list
    time_index_list = []

    for i in range(len(time_list)):
        single_time_floor = time_list_floor[i]
        single_time_ceil = time_list_ceil[i]
        time_index = np.where(data_time_list == single_time_ceil)[0]
        time_index_2 = np.where(data_time_list == single_time_floor)[0]
        time_index_list.extend(time_index)
        time_index_list.extend(time_index_2)

    # Sort the list in ascending order
    time_index_list = np.sort(time_index_list)

    for i in range(len(time_index_list)):

        # Plot the image at the time index
        fig_cursor = plot_stat_map(index_img(img_cursor, time_index_list[i]), t1_fname, threshold=threshold,
                                   title='[Cursor] ' + 'Time = ' + str(time_list_floor[i]) + ' ms')

        fig_robot = plot_stat_map(index_img(img_robot, time_index_list[i]), t1_fname, threshold=threshold,
                                  title='[Robot] ' + 'Time = ' + str(time_list_floor[i]) + ' ms')

        if save is True:
            # Change to the save directory
            os.chdir(save_directory)

            # Save the figure
            fig_cursor.savefig('peak_activation_volume_cursor_' + str(time_list_floor[i]) + '_ms')
            fig_robot.savefig('peak_activation_volume_robot_' + str(time_list_floor[i]) + '_ms')

            # Change back to original directory
            os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')
    pass


def visualize_cluster(clu, cz_data_cache, parameter_cache, modelling_parameters, subjects_dir):
    """Plot and saves the clusters (statistical data).

    :param clu: Obtained from perform_statistics function
    :param cz_data_cache: Obtained from the obtain_cz_cache function
    :param parameter_cache: Obtained from morph_data function
    :param modelling_parameters: The parameters for the modelling
    :param subjects_dir: Directory to the brain model
    :return: -
    """
    # Obtain current directory
    current_dir = os.getcwd()
    save_directory = current_dir + '/Saved_Images/' + 'grand_average/' + 'statistics'

    if not os.path.exists(save_directory):
        if not os.path.exists(current_dir + '/Saved_Images/grand_average'):
            if not os.path.exists(current_dir + '/Saved_Images'):
                os.mkdir(current_dir + '/Saved_Images')
            os.mkdir(current_dir + '/Saved_Images/grand_average')
        # Make save directory
        os.mkdir(save_directory)

    # Unpack the cz_cache and append to time list
    time_list = []
    peak_data_time = cz_data_cache['peak_data_time']
    min_data_time = cz_data_cache['min_data_time']
    time_list.extend(peak_data_time)
    time_list.append(min_data_time)
    time_list = np.sort(time_list)

    # Unpack the parameter_cache
    t_step = parameter_cache['t_step']
    fs_ave_vertices = parameter_cache['fs_ave_vertices']

    # Unpack the modelling_cache
    hemisphere = modelling_parameters['hemisphere']
    selected_view = modelling_parameters['view']
    vector = modelling_parameters['vector']
    epsilon = modelling_parameters['epsilon']
    save = modelling_parameters['save']

    # Set the number of rows and columns
    ncol = len(time_list)

    # Set the figure size
    fig = plt.figure(figsize=(48, 12))

    for i in range(len(time_list)):

        plt.subplot(1, ncol, i + 1)

        # Build a convenient representation of each cluster, where each cluster becomes a "time point" in the SourceEstimate
        stc_all_cluster = summarize_clusters_stc(clu, tstep=t_step, vertices=fs_ave_vertices, subject='fsaverage')

        # Save the num of vertices and time points
        n_vertices, n_time_pts = stc_all_cluster.shape

        # Set the points from time point 1 and above to be 0
        stc_all_cluster.data[:, 1:] = 0

        # Set the min time value
        time_min = time_list[i] - epsilon[str(i + 1) + '_min']

        # Set the max time value
        time_max = time_list[i] + epsilon[str(i + 1) + '_max']

        for sin_ver in range(n_vertices):
            if stc_all_cluster.data[sin_ver, 0] < time_min:
                stc_all_cluster.data[sin_ver, 0] = 0
            if stc_all_cluster.data[sin_ver, 0] > time_max:
                stc_all_cluster.data[sin_ver, 0] = 0
            if stc_all_cluster.data[sin_ver, 0] != 0:
                stc_all_cluster.data[sin_ver, 0] = time_list[i]

        time_min = round(time_min)
        time_max = round(time_max)
        time_point = round(time_list[i])

        # Set the surfer_kwargs
        surfer_kwargs = dict(hemi=hemisphere, subjects_dir=subjects_dir, colorbar=False,
                             clim=dict(kind='value', lims=[time_min, time_point, time_max]), views=selected_view,
                             size=(800, 800), smoothing_steps=5, time_label='Vertices with statistical differences')

        # Blue blobs are for condition A < condition B, red for A > B
        model = stc_all_cluster.plot(**surfer_kwargs)
        # model = stc_all_cluster.plot()
        model.add_text(0.1, 0.9, 'Time Range: ' + str(time_min) + ' ms to ' + str(time_max) + ' ms', 'title',
                       font_size=36)
        single_screenshot = model.screenshot_single()
        plt.imshow(single_screenshot)
        fig.suptitle(
            'Statistical Differences ' + '[' + hemisphere.upper() + '] (' + selected_view.upper() + ' VIEW) ' + '[' + method + ']',
            fontsize=54)

        if save is True:
            # Change to the save directory
            os.chdir(save_directory)

            # Save the images
            model.save_image(
                'Statistical_' + hemisphere.upper() + '_' + selected_view.upper() + '_' + str(time_min) + '_to_' + str(
                    time_max) + '.png')

            plt.savefig(
                'Statistical_' + hemisphere.upper() + '_' + selected_view.upper() + '_statistical_time_montage.png')

            # Change back to original directory
            os.chdir('/home/sailam/Desktop/MSNE/Research Project (9 Weeks)/Data/dataset-ErrP-HRI-master')
    pass
