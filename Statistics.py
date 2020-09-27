import numpy as np
from scipy import stats as stats

import mne
from mne import (io, spatio_temporal_tris_connectivity, spatial_tris_connectivity, compute_morph_matrix, grade_to_tris)
from mne.stats import (spatio_temporal_cluster_1samp_test, summarize_clusters_stc)

from Modelling import *


def obtain_statistical_data(data_cursor, data_robot, subject_to_process, method, vector, trans, src, bem):
    """Obtains the data required to perform statistics.

    :param data_cursor: -
    :param data_robot: -
    :param subject_to_process: The subject list to process
    :param method: Method to perform modelling ('sLORETA' etc.)
    :param vector: Specify to show in vector form (True or False)
    :param trans: Model obtained from get_FS_data function
    :param src: Model obtained from get_FS_data function
    :param bem: Model obtained from get_FS_data function
    :return: data_array, parameters_cache
    """
    # Perform modelling for 1 subject
    model_data, model_cache = perform_modelling(data_cursor['s02'], method, trans, src, bem)
    n_vertices, n_vecs, n_times = model_data['contrast_stc_vec'].shape

    # Initialization the empty arrays
    contrast_cursor_array = np.array([]).reshape(n_vertices, n_times, 0)
    contrast_robot_array = np.array([]).reshape(n_vertices, n_times, 0)

    contrast_vec_cursor_array = np.array([]).reshape(n_vertices, n_vecs, n_times, 0)
    contrast_vec_robot_array = np.array([]).reshape(n_vertices, n_vecs, n_times, 0)

    for i in range(len(subject_to_process)):

        # Perform modelling for single subject
        single_subject = subject_to_process[i]
        print('\nProcessing for subject: ' + single_subject + '\n')

        modelling_data_cursor, model_cache_cursor = perform_modelling(data_cursor[single_subject], method, trans, src,
                                                                      bem)
        modelling_data_robot, model_cache_robot = perform_modelling(data_robot[single_subject], method, trans, src, bem)

        if vector is False:

            # Load the data
            contrast_cursor = modelling_data_cursor['contrast_stc'].data  ### Shape is (20484, 257)
            contrast_robot = modelling_data_robot['contrast_stc'].data

            # Concatenate the data for all subjects
            contrast_cursor_array = np.concatenate((contrast_cursor_array, contrast_cursor[:, :, np.newaxis]), axis=2)
            contrast_robot_array = np.concatenate((contrast_robot_array, contrast_robot[:, :, np.newaxis]), axis=2)

        else:

            # Load the data
            contrast_vec_cursor = modelling_data_cursor['contrast_stc_vec'].shape
            contrast_vec_robot = modelling_data_robot['contrast_stc_vec'].shape

            # Concatenate the data for all subjects
            contrast_vec_cursor_array = np.concatenate(
                (contrast_vec_cursor_array, contrast_vec_cursor[:, :, :, np.newaxis]), axis=3)
            contrast_vec_robot_array = np.concatenate(
                (contrast_vec_robot_array, contrast_vec_robot[:, :, :, np.newaxis]), axis=3)

    if vector is False:

        # Concatenate the data for cursor and robot
        data_array = np.concatenate(
            (contrast_cursor_array[:, :, :, np.newaxis], contrast_robot_array[:, :, :, np.newaxis]), axis=3)

        # Delete unnecessary data
        del contrast_cursor_array, contrast_robot_array

        # Unpack the inverse operator
        inverse_operator = model_cache_cursor['inv']
        subject_vertices = [s['vertno'] for s in inverse_operator['src']]

        # Total number of subjects
        n_subjects = len(subject_to_process)

        # Cache the parameters in dictionary (for 1 subject, which is representative of both cursor and robot)
        t_step = modelling_data_cursor['contrast_stc'].tstep
        n_vertices, n_times = modelling_data_cursor['contrast_stc'].data.shape
        parameters_cache = {'t_step': t_step, 'n_vertices': n_vertices, 'n_times': n_times, 'n_subjects': n_subjects,
                            'subject_vertices': subject_vertices}

    else:
        # Concatenate the data for cursor and robot
        data_array = np.concatenate(
            (contrast_vec_cursor_array[:, :, :, :, np.newaxis], contrast_vec_robot_array[:, :, :, :, np.newaxis]),
            axis=4)

        # Delete unnecessary data
        del contrast_vec_cursor_array, contrast_vec_robot_array

        # Unpack the inverse operator
        inverse_operator = model_cache['inv']
        subject_vertices = [s['vertno'] for s in inverse_operator['src']]

        # Total number of subjects
        n_subjects = len(subject_to_process)

        # Cache the parameters in dictionary (for 1 subject, which is representative of both cursor and robot)
        t_step = modelling_data_cursor['contrast_stc_vec'].step
        n_vertices, n_vec, n_times = modelling_data_cursor['contrast_stc_vec'].data.shape
        parameters_cache = {'t_step': t_step, 'n_vertices': n_vertices, 'n_vec': n_vec, 'n_times': n_times,
                            'n_subjects': n_subjects, 'subject_vertices': subject_vertices}
    return data_array, parameters_cache


def morph_data(data_array, parameters_cache, vector, subjects_dir):
    """Morphs the subject brains to fs_average brain.

    :param data_array: Statistical data obtained from obtain_statistical_data function
    :param parameters_cache: Statistical parameters cache obtained from obtain_statistical_data function
    :param vector: Method to perform modelling ('sLORETA' etc.)
    :param subjects_dir: Directory to the brain model
    :return:  (morphed brain data array)
    """
    # Unpack parameter cache dictionary
    n_vertices = parameters_cache['n_vertices']
    n_times = parameters_cache['n_times']
    n_subjects = parameters_cache['n_subjects']
    subject_vertices = parameters_cache['subject_vertices']

    if vector is True:
        n_vec = parameters_cache['n_vec']

    # Create the fs average vertices
    fs_ave_vertices = [np.arange(10242), np.arange(10242)]

    # Add in fs_ave_vertices to parameter_cache
    parameters_cache['fs_ave_vertices'] = fs_ave_vertices

    # Compute the morph matrix
    smooth_int = 20
    morph_mat = compute_morph_matrix('subjects', 'fsaverage', subject_vertices, fs_ave_vertices, smooth_int,
                                     subjects_dir)
    n_vertices_fs_ave = morph_mat.shape[0]
    # morph_mat shape is (20484, 20484)

    # Reshape in order for dot() to work properly
    if vector is False:
        X = data_array.reshape(n_vertices, n_times * n_subjects * 2)  # Shape is (20484, 257*12*2)
    else:
        X = data_array.reshape(n_vertices * n_vec,
                               n_times * n_subjects * 2)  # Shape is (20484*3, 257*12*2) ##### TO DOUBLE CHECK #####

    print('Morphing data...')

    X = morph_mat.dot(X)  # morph_mat is a sparse matrix

    # Reshape into (vertices, times, subjects and conditions)
    if vector is False:
        X = X.reshape(n_vertices_fs_ave, n_times, n_subjects, 2)  # Shape is (20484, 257, 12, 2)
    # Reshape into (vertices, vecs, times, subjects and conditions)
    else:
        X = X.reshape(n_vertices, n_vec, n_times, n_subjects, 2)  # Shape is (20484, 3, 257, 12, 2)
    return X, parameters_cache


def perform_statistics(morphed_data, parameter_cache, vector, p_value=None):
    """Performs the statistical analysis using spatial_tris_connectivity.

    :param morphed_data: Morphed data obtained from morph_data function
    :param parameter_cache: Morphed parameter cache obtained from morph_data function.
    :param vector: Method to perform modelling ('sLORETA' etc.)
    :param p_value: Statistical p-value
    :return: clu, good_cluster_inds
    """
    # Unpack parameter cache dictionary
    n_subjects = parameter_cache['n_subjects']
    n_times = parameter_cache['n_times']

    # Take on the absolute
    X = np.abs(morphed_data)

    # Obtain the paired contrast
    if vector is False:
        X = X[:, :, :, 0] - X[:, :, :, 1]  # Dimension is (space, time, subjects)
    else:
        X = X[:, :, :, :, 0] - X[:, :, :, :, 1]  # Dimension is (space, vector, time, subjects)

    print('Computing connectivity... ')
    connectivity = spatial_tris_connectivity(grade_to_tris(5))

    # Note that X needs to be a multi-dimensional array of shape [samples (subjects) x time x space]
    if vector is False:
        X = np.transpose(X, [2, 1, 0])
    else:
        X = np.transpose(X, [3, 2, 1, 0])  ##### TO DOUBLE CHECK #####

    # Perform the clustering
    p_threshold = p_value  # 0.001
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)

    print('Clustering... ')
    T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(X, connectivity=connectivity, n_jobs=1,
                                                                               threshold=t_threshold)

    # Pack the outputs into tuple
    clu = (T_obs, clusters, cluster_p_values, H0)

    # Select the clusters that are sig. at p < p_value (Note this value is multiple-comparisons corrected)
    good_cluster_inds = np.where(cluster_p_values < p_value)[0]
    return clu, good_cluster_inds


def perform_statistics_2(morphed_data, parameter_cache, vector, p_value=None):
    """Performs the statistical analysis using spatial_tris_connectivity.

    :param morphed_data: Morphed data obtained from morph_data function
    :param parameter_cache: Morphed parameter cache obtained from morph_data function.
    :param vector: Method to perform modelling ('sLORETA' etc.)
    :param p_value: Statistical p-value
    :return: clu, good_cluster_inds
    """
    # Unpack parameter cache dictionary
    n_subjects = parameter_cache['n_subjects']
    n_times = parameter_cache['n_times']

    # Take on the absolute
    X = np.abs(morphed_data)

    # Obtain the paired contrast
    if vector is False:
        X = X[:, :, :, 0] - X[:, :, :, 1]  # Dimension is (space, time, subjects)
    else:
        X = X[:, :, :, :, 0] - X[:, :, :, :, 1]  # Dimension is (space, vector, time, subjects)

    print('Computing connectivity... ')
    connectivity_2 = mne.spatio_temporal_tris_connectivity(grade_to_tris(5), n_times)

    # Note that X needs to be a multi-dimensional array of shape [samples (subjects) x time x space]
    if vector is False:
        X = np.transpose(X, [2, 1, 0])
    else:
        X = np.transpose(X, [3, 2, 1, 0])  ##### TO DOUBLE CHECK #####

    # Perform the clustering
    p_threshold = p_value  # 0.001
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)

    print('Clustering... ')
    T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(X, connectivity=connectivity_2, n_jobs=1,
                                                                               threshold=t_threshold)

    # Pack the outputs into tuple
    clu = (T_obs, clusters, cluster_p_values, H0)

    # Select the clusters that are sig. at p < p_value (Note this value is multiple-comparisons corrected)
    good_cluster_inds = np.where(cluster_p_values < p_value)[0]
    return clu, good_cluster_inds