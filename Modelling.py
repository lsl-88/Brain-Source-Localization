import numpy as np

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse


def compute_grand_average(data, subject_to_process):
    """Computes the grand average across all the subjects.

    :param data: Respective cursor or robot dataset
    :param subject_to_process: The subject list to process
    :return: grand_average_cache (dict)
    """
    # Initialize empty list
    epo_nE_list = []
    epo_Er_list = []
    contrast_list = []
    cov_list = []

    for i in range(len(subject_to_process)):
        # Single subject
        single_subject = subject_to_process[i]

        epo_nE_avg = data[single_subject]['epo_nE_avg']
        epo_Er_avg = data[single_subject]['epo_Er_avg']
        contrast = data[single_subject]['contrast']
        cov = data[single_subject]['cov']

        # Append to list
        epo_nE_list.append(epo_nE_avg)
        epo_Er_list.append(epo_Er_avg)
        contrast_list.append(contrast)
        cov_list.append(cov)

    # Initialize empty list for cov data and loglikelihood
    cov_data_list = []
    logll_list = []

    for i in range(len(cov_list)):
        single_cov = cov_list[i]['data']
        single_loglik = cov_list[i]['loglik']

        cov_data_list.append(single_cov)
        logll_list.append(single_loglik)

    # Stack them together
    total_cov = np.stack(cov_data_list, axis=2)
    total_logll = np.stack(logll_list, axis=0)

    # Compute average for cov data and loglikelihood
    avg_cov = np.mean(total_cov, axis=2)
    avg_logll = np.mean(total_logll, axis=0)

    # Update the values manually
    single_cov = cov_list[0]
    single_cov['data'] = avg_cov
    single_cov['loglik'] = avg_logll

    # Compute grand average
    epo_nE_grand_avg = mne.grand_average(epo_nE_list)
    epo_Er_grand_avg = mne.grand_average(epo_Er_list)
    contrast_grand_average = mne.grand_average(contrast_list)

    # Pack into dictionary
    grand_average_cache = {'cov': single_cov, 'epo_nE_grand_avg': epo_nE_grand_avg, 'epo_Er_grand_avg': epo_Er_grand_avg,
                           'contrast_grand_average': contrast_grand_average}
    return grand_average_cache


def perform_modelling(data, method, trans, src, bem):
    """Performs the forward and inverse modelling and return the data.

    :param data: Cursor or robot data
    :param method: Method to perform modelling ('sLORETA' etc.)
    :param trans: Model obtained from get_FS_data function
    :param src: Model obtained from get_FS_data function
    :param bem: Model obtained from get_FS_data function
    :return: modelling_data, model_cache
    """
    # Unpack the data
    epo = data['epo']
    epo_nE_avg = data['epo_nE_avg']
    epo_Er_avg = data['epo_Er_avg']
    cov = data['cov']
    contrast = data['contrast']

    # Perform Forward Modelling
    fwd = mne.make_forward_solution(epo.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=1)

    # Assemble inverse operator
    info = epo.info
    inv = make_inverse_operator(info, fwd, cov, loose=0.2, depth=0.8)

    # Apply inverse operator to evoked data
    if method is None:
        method = 'dSPM' #'dSPM' 'MNE' 'sLORETA' 'eLORETA'

    contrast_stc = apply_inverse(contrast, inv, lambda2=1. / 9., method=method)
    contrast_stc_vec = apply_inverse(contrast, inv, lambda2=1. / 9., method=method, pick_ori='vector')

    nonErr_stc = apply_inverse(epo_nE_avg, inv, lambda2=1. / 9., method=method)
    nonErr_stc_vec = apply_inverse(epo_nE_avg, inv, lambda2=1. / 9., method=method, pick_ori='vector')

    Err_stc = apply_inverse(epo_Er_avg, inv, lambda2=1. / 9., method=method)
    Err_stc_vec = apply_inverse(epo_Er_avg, inv, lambda2=1. / 9., method=method, pick_ori='vector')

    # Cache the data
    modelling_data = {'contrast_stc': contrast_stc, 'contrast_stc_vec': contrast_stc_vec,
                      'nonErr_stc': nonErr_stc, 'nonErr_stc_vec': nonErr_stc_vec,
                      'Err_stc': Err_stc, 'Err_stc_vec': Err_stc_vec}

    model_cache = {'fwd': fwd, 'inv': inv}
    return modelling_data, model_cache