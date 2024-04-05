import pathlib
import mne
import csv
import numpy as np
from mne_bids import BIDSPath, get_entity_vals, find_matching_paths
from scipy.fft import fft, ifft
import random
import math
import matplotlib.pyplot as plt
mne.set_log_level('warning')
import warnings
from scipy.signal import welch
from scipy.stats import iqr, skew, kurtosis
import cmath
import tensorflow
warnings.filterwarnings("ignore")
import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def calculate_psd_relative_power(eeg_segment):
    frequencies, psd = welch(eeg_segment, fs, nperseg=256)

    # Calculate relative power in specific frequency bands
    total_power = np.sum(psd)
    delta_power = np.sum(psd[(frequencies >= delta_band[0]) & (frequencies < delta_band[1])])
    theta_power = np.sum(psd[(frequencies >= theta_band[0]) & (frequencies < theta_band[1])])
    alpha_power = np.sum(psd[(frequencies >= alpha_band[0]) & (frequencies < alpha_band[1])])
    beta_power = np.sum(psd[(frequencies >= beta_band[0]) & (frequencies < beta_band[1])])
    gamma_power = np.sum(psd[(frequencies >= gamma_band[0]) & (frequencies < gamma_band[1])])

    # Calculate relative power ratios
    delta_theta_ratio = delta_power / theta_power
    theta_alpha_ratio = theta_power / alpha_power

    return {
        'delta_power': delta_power / total_power,
        'theta_power': theta_power / total_power,
        'alpha_power': alpha_power / total_power,
        'beta_power': beta_power / total_power,
        'gamma_power': gamma_power / total_power,
        'delta_theta_ratio': delta_theta_ratio,
        'theta_alpha_ratio': theta_alpha_ratio
    }

total_subjects = 88 
num_subjects = 88 #How many subjects to include in analysis
random_subjects = False # Set this boolean to True to read subjects randomly
segment_data = False  # Set this boolean to True to segment the data
segment_duration = 30  # Duration for segmenting the EEG data (in seconds)
num_measures= 19
total_measures = 3*num_measures+7
delta_band = (0.5, 4)
theta_band = (4, 8)
alpha_band = (8, 14)
beta_band = (14, 30)
gamma_band = (30, 45)
fs = 500
arr = np.empty((0, 5))
method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
# Load participant data into 'arr' from a TSV file
with open("AD_dataset/participants.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        arr = np.append(arr, np.array([row]), axis=0)



# Initialize lists to store segmented EEG data for each group (A, C, and F)
a_segments, c_segments, f_segments = [], [], []

# Root directory for the BIDS dataset
bids_root = pathlib.Path('AD_dataset/derivatives')

# Get available session values from the BIDS dataset
sessions = get_entity_vals(bids_root, 'session', ignore_sessions='on')

# Set the data type for BIDS search (EEG data)
datatype = 'eeg'

# File extensions to consider (ignore .json files)
extensions = [".bdf", ".tsv"]

# Find BIDS paths for the EEG data matching the specified criteria
bids_paths = find_matching_paths(bids_root, datatypes=datatype, sessions=sessions, extensions=extensions)

# Variables for averaging the scores in each group
asum, anum = 0, 0
csum, cnum = 0, 0
fsum, fnum = 0, 0


# Task name for the BIDS dataset
task = 'eyesclosed'

# Suffix for the BIDS dataset
suffix = 'eeg'

# Generate a list of subject indices (1 to total_subjects)
subject_indices = list(range(1, total_subjects))

# If random_subjects is True, shuffle the subject indices
if random_subjects:
    random.shuffle(subject_indices)

# Select a subset of subject indices for processing
subject_indices = subject_indices[:num_subjects]

shortest_length = float('inf')  # Initialize with infinity
subj_idx = subject_indices[1]
subject = f'00{subj_idx}' if subj_idx <= 9 else f'0{subj_idx}'

# Define the BIDS path for the subject's EEG data
bids_path = BIDSPath(root=bids_root, subject=subject, datatype=datatype)
bids_path = bids_path.update(task=task, suffix=suffix)

# Read the raw EEG data for the subject
raw = mne.io.read_raw_eeglab(bids_path, verbose=False)

# Get the total duration of the EEG recording
total_duration = raw.times[-1]
start_time = 0
end_time = segment_duration
segment = raw.copy().crop(tmin=start_time, tmax=end_time)


import mne
from pathlib import Path
raw_short = raw.copy().crop(tmin=0, tmax=segment_duration)

# Load the MNE sample data to obtain MRI information
subjects_dir = r'C:\Users\mikem\mne_data\MNE-fsaverage-data'

trans_fname =  r'AD_dataset/ad-icp-trans.fif'
montage = mne.channels.make_standard_montage('standard_1020')
raw_short.set_montage(montage)

#mne.gui.coregistration(subjects_dir=subjects_dir, subject='fsaverage', inst=r'AD_dataset/epochs_for_source_epo.fif')

#fig = mne.viz.plot_alignment(info=raw_short.info, trans=trans_fname, subject='fsaverage', dig=True, subjects_dir=subjects_dir, verbose=True)

subject = 'fsaverage' # Use oct6 during an actual analysis! 
src = mne.setup_source_space(subject=subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)  # Remove this one during an actual analysis!
src
#mne.viz.plot_alignment(info=raw_short.info, trans=trans_fname, subject=subject, src=src, subjects_dir=subjects_dir, dig=True, surfaces=['head-dense', 'white'], coord_frame='meg')

#conductivity = (0.3,)  # for single layer – used in MEG
conductivity = (0.3, 0.006, 0.3)  # for three layers – used in EEG
model = mne.make_bem_model(subject=subject, ico=2,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
model
bem_sol = mne.make_bem_solution(model)
bem_sol
bem_fname = 'sample_bem.fif'
mne.bem.write_bem_solution(bem_fname, bem_sol, overwrite=True)

fwd = mne.make_forward_solution(raw_short.info, trans=trans_fname, src=src, bem=bem_sol,meg=False, eeg=True, mindist=5.0, n_jobs=1) 
fwd

labels = mne.read_labels_from_annot(subject, parc="aparc", subjects_dir=subjects_dir)

results = np.zeros((num_subjects, total_measures, len(labels)))
total_segments = [0,0,0]
# Initialize the results_avg matrix with zeros
# The third dimension is 3 to store averages for each group (A, C, F)
results_avg = np.zeros((3, 2, len(labels)))
results_calc = np.zeros((3, 2, len(labels)))
# Iterate over subjects
subj_groups = []
measures = ["max", "mean", "median", "inter-quartile range", "range", "25th percentile", "75th percentile", "coefficient of vatiation", "signal energy", "entropy", "kurtosis", "variance", "standard deviation", "min value", "skewness", "autocorrelation_mean", "autocorrelation_zero_lag", "autocorrelation_std", "autocorrelatiion_max", "max BETA", "mean BETA", "median BETA", "inter-quartile range BETA", "range BETA", "25th percentile BETA", "75th percentile BETA", "coefficient of vatiation BETA", "signal energy BETA", "entropy BETA", "kurtosis BETA", "variance BETA", "standard deviation BETA", "min value BETA", "skewness BETA", "autocorrelation_mean BETA", "autocorrelation_zero_lag BETA", "autocorrelation_std BETA", "autocorrelatiion_max BETA", "max GAMMA", "mean GAMMA", "median GAMMA", "inter-quartile range GAMMA", "range GAMMA", "25th percentile GAMMA", "75th percentile GAMMA", "coefficient of vatiation GAMMA", "signal energy GAMMA", "entropy GAMMA", "kurtosis GAMMA", "variance GAMMA", "standard deviation GAMMA", "min value GAMMA", "skewness GAMMA", "autocorrelation_mean GAMMA", "autocorrelation_zero_lag GAMMA", "autocorrelation_std GAMMA", "autocorrelatiion_max GAMMA", "delta power", "theta power", "alpha power", "beta power", "gamma power", "delta-theta ratio", "theta-alpha ratio"]      

for subj_idx in subject_indices:
    # Generate subject IDs in the format "001" to "088"
    subject = f'00{subj_idx}' if subj_idx <= 9 else f'0{subj_idx}'
    print(subj_idx)
    # Define the BIDS path for the subject's EEG data
    bids_path = BIDSPath(root=bids_root, subject=subject, datatype=datatype)
    bids_path = bids_path.update(task=task, suffix=suffix)

    # Read the raw EEG data for the subject
    raw = mne.io.read_raw_eeglab(bids_path, verbose=False)

    # Get the total duration of the EEG recording
    total_duration = raw.times[-1]

    # Get the group (A, C, or F) and score for the subject from the 'arr' array
    group = arr[subj_idx][3]
    score = arr[subj_idx][4]
    print(group)
    subj_groups.append(group)
    noise_cov = mne.compute_raw_covariance(raw)
    noise_cov = mne.cov.regularize(noise_cov, raw.info, mag=0.1, grad=0.1, eeg=0.1, proj=True)
    noise_cov
    #mne.viz.plot_cov(noise_cov, info=raw.info)
    #plt.figure()
    #plt.imshow(noise_cov.data, cmap='viridis', origin='lower')
    #plt.title('Noise Covariance Matrix')
    #plt.colorbar()
    #plt.show()
    segments = []
    if group == "A":
        avg_idx = 2
    elif group == "C":
        avg_idx = 0
    elif group == "F":
        avg_idx = 1
    else:
        raise ValueError("Invalid group")
    num_segments = int(total_duration // segment_duration)
    for ii in range(num_segments):
        start_time = ii * segment_duration
        end_time = (ii + 1) * segment_duration
        segment = raw.copy().crop(tmin=start_time, tmax=end_time)
        segments.append(segment)   
    i = 0    
    for segment in segments:
        for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
                                  locals().items())), key= lambda x: -x[1])[:10]:
            print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))   
        segment.set_eeg_reference(projection=True)
        inverse_operator = mne.minimum_norm.make_inverse_operator(segment.info, fwd, noise_cov, loose=0.2, depth=0.8)
        

        
        stc = mne.minimum_norm.apply_inverse_raw(segment, inverse_operator, lambda2, method=method, verbose=True)

        #brain = stc.plot(surface='pial',hemi='both',subjects_dir=subjects_dir,time_viewer=True)

        stc_mean = stc.mean()

        ii=0
        for label in labels:
            try: 
            # if label.hemi == 'lh':
                #     hemisphere_color = "k"
                # elif label.hemi == 'rh':
                #     hemisphere_color = "r"
                # load the anatomical ROI for comparison
                
                # extract the anatomical time course for each label
                stc_anat_label = stc.in_label(label)
                pca_anat = stc.extract_label_time_course(label, src, mode="pca_flip")[0]
        
                pca_anat *= np.sign(pca_anat[np.argmax(np.abs(pca_anat))])
        
                # plt.figure()
                # plt.plot(
                #     1e3 * stc_anat_label.times, pca_anat, "k", label="Anatomical %s" % label.name
                # )
                # plt.legend()
                # plt.show()
                mean_label=np.mean(pca_anat, axis=0)
                max_label = np.max(np.abs(pca_anat), axis=0)
                percentiles = np.percentile(pca_anat, [25, 75], axis=0)
                std_deviation_label = np.std(pca_anat, axis=0)
                cv_label = std_deviation_label / mean_label  # Coefficient of Variation
                signal_energy_label = np.sum(pca_anat ** 2, axis=0)
                entropy_label = -np.sum(pca_anat * np.log(pca_anat), axis=0)
                autocorrelation = np.correlate(pca_anat, pca_anat, mode='full')
                autocorrelation_max = max(autocorrelation)
                autocorrelation_zero_lag = autocorrelation[len(autocorrelation) // 2]
                autocorrelation_std  = np.std(autocorrelation)
                autocorrelation_mean = np.mean(autocorrelation)
                variance_label = np.var(pca_anat, axis=0)
                min_value_label = np.min(pca_anat, axis=0)
                skewness_label = skew(pca_anat, axis=0)
                kurtosis_label = kurtosis(pca_anat, axis=0)
                median_label = np.median(pca_anat, axis=0)
                iqr_label = np.percentile(pca_anat, 75, axis=0) - np.percentile(pca_anat, 25, axis=0)
                range_label = np.max(pca_anat, axis=0) - np.min(pca_anat, axis=0)
                
                
                results[subj_idx-1][0][ii] = results[subj_idx-1][0][ii]+ max_label#
                results[subj_idx-1][1][ii] = results[subj_idx-1][1][ii]+ mean_label
                results[subj_idx-1][2][ii] = results[subj_idx-1][2][ii] + median_label
                results[subj_idx-1][3][ii] = results[subj_idx-1][3][ii] + iqr_label
                results[subj_idx-1][4][ii] = results[subj_idx-1][4][ii] + range_label#
                results[subj_idx-1][5][ii] = results[subj_idx-1][5][ii] + percentiles[0]  # 25th percentile
                results[subj_idx-1][6][ii] = results[subj_idx-1][6][ii] + percentiles[1]  # 75th percentile
                results[subj_idx-1][7][ii] = results[subj_idx-1][7][ii] + cv_label#
                results[subj_idx-1][8][ii] = results[subj_idx-1][8][ii] + signal_energy_label
                results[subj_idx-1][9][ii] = results[subj_idx-1][9][ii] + entropy_label#
                results[subj_idx-1][10][ii] = results[subj_idx-1][10][ii] + kurtosis_label#
                results[subj_idx-1][11][ii] = results[subj_idx-1][11][ii] + variance_label
                results[subj_idx-1][12][ii] = results[subj_idx-1][12][ii] + std_deviation_label
                results[subj_idx-1][13][ii] = results[subj_idx-1][13][ii] + min_value_label#
                results[subj_idx-1][14][ii] = results[subj_idx-1][14][ii] + skewness_label#
                results[subj_idx-1][15][ii] = results[subj_idx-1][15][ii] + autocorrelation_mean#?
                results[subj_idx-1][16][ii] = results[subj_idx-1][16][ii] + autocorrelation_zero_lag#!
                results[subj_idx-1][17][ii] = results[subj_idx-1][17][ii] + autocorrelation_std#!
                results[subj_idx-1][18][ii] = results[subj_idx-1][18][ii] + autocorrelation_max
                
                y = fft(pca_anat)
                N = len(pca_anat)
                delta_f = 500 / N
                lower_bin_beta = int(beta_band[0] / delta_f)
                upper_bin_beta = int(beta_band[1] / delta_f)
                lower_bin_gamma = int(gamma_band[0] / delta_f)
                upper_bin_gamma = int(gamma_band[1] / delta_f)
                freq_beta = y[lower_bin_beta:upper_bin_beta]
                beta = ifft(freq_beta).real
                freq_gamma = y[lower_bin_gamma:upper_bin_gamma]
                gamma = ifft(freq_gamma).real
                
                
                mean_label=np.mean(beta, axis=0)
                max_label = np.max(np.abs(beta), axis=0)
                percentiles = np.percentile(beta, [25, 75], axis=0)
                std_deviation_label = np.std(beta, axis=0)
                cv_label = std_deviation_label / mean_label  # Coefficient of Variation
                signal_energy_label = np.sum(beta ** 2, axis=0)
                entropy_label = -np.sum(beta * np.log(beta), axis=0)
                autocorrelation = np.correlate(beta, beta, mode='full')
                autocorrelation_max = max(autocorrelation)
                autocorrelation_zero_lag = autocorrelation[len(autocorrelation) // 2]
                autocorrelation_std  = np.std(autocorrelation)
                autocorrelation_mean = np.mean(autocorrelation)
                variance_label = np.var(beta, axis=0)
                min_value_label = np.min(beta, axis=0)
                skewness_label = skew(beta, axis=0)
                kurtosis_label = kurtosis(beta, axis=0)
                median_label = np.median(beta, axis=0)
                iqr_label = np.percentile(beta, 75, axis=0) - np.percentile(beta, 25, axis=0)
                range_label = np.max(beta, axis=0) - np.min(beta, axis=0)
                
                
                results[subj_idx-1][0+num_measures][ii] = results[subj_idx-1][0+num_measures][ii]+ max_label#!
                results[subj_idx-1][1+num_measures][ii] = results[subj_idx-1][1+num_measures][ii]+ mean_label#
                results[subj_idx-1][2+num_measures][ii] = results[subj_idx-1][2+num_measures][ii] + median_label#
                results[subj_idx-1][3+num_measures][ii] = results[subj_idx-1][3+num_measures][ii] + iqr_label#!
                results[subj_idx-1][4+num_measures][ii] = results[subj_idx-1][4+num_measures][ii] + range_label#!
                results[subj_idx-1][5+num_measures][ii] = results[subj_idx-1][5+num_measures][ii] + percentiles[0]  # 25th percentile#!
                results[subj_idx-1][6+num_measures][ii] = results[subj_idx-1][6+num_measures][ii] + percentiles[1]  # 75th percentile#!!
                results[subj_idx-1][7+num_measures][ii] = results[subj_idx-1][7+num_measures][ii] + cv_label
                results[subj_idx-1][8+num_measures][ii] = results[subj_idx-1][8+num_measures][ii] + signal_energy_label#!!
                results[subj_idx-1][9+num_measures][ii] = results[subj_idx-1][9+num_measures][ii] + entropy_label
                results[subj_idx-1][10+num_measures][ii] = results[subj_idx-1][10+num_measures][ii] + kurtosis_label##outliers
                results[subj_idx-1][11+num_measures][ii] = results[subj_idx-1][11+num_measures][ii] + variance_label#!
                results[subj_idx-1][12+num_measures][ii] = results[subj_idx-1][12+num_measures][ii] + std_deviation_label#!!
                results[subj_idx-1][13+num_measures][ii] = results[subj_idx-1][13+num_measures][ii] + min_value_label#!
                results[subj_idx-1][14+num_measures][ii] = results[subj_idx-1][14+num_measures][ii] + skewness_label
                results[subj_idx-1][15+num_measures][ii] = results[subj_idx-1][15+num_measures][ii] + autocorrelation_mean
                results[subj_idx-1][16+num_measures][ii] = results[subj_idx-1][16+num_measures][ii] + autocorrelation_zero_lag#!!!
                results[subj_idx-1][17+num_measures][ii] = results[subj_idx-1][17+num_measures][ii] + autocorrelation_std#!!
                results[subj_idx-1][18+num_measures][ii] = results[subj_idx-1][18+num_measures][ii] + autocorrelation_max#!
                
                mean_label=np.mean(gamma, axis=0)
                max_label = np.max(np.abs(gamma), axis=0)
                percentiles = np.percentile(gamma, [25, 75], axis=0)
                std_deviation_label = np.std(gamma, axis=0)
                cv_label = std_deviation_label / mean_label  # Coefficient of Variation
                signal_energy_label = np.sum(gamma ** 2, axis=0)
                entropy_label = -np.sum(gamma * np.log(gamma), axis=0)
                autocorrelation = np.correlate(gamma, gamma, mode='full')
                autocorrelation_max = max(autocorrelation)
                autocorrelation_zero_lag = autocorrelation[len(autocorrelation) // 2]
                autocorrelation_std  = np.std(autocorrelation)
                autocorrelation_mean = np.mean(autocorrelation)
                variance_label = np.var(gamma, axis=0)
                min_value_label = np.min(gamma, axis=0)
                skewness_label = skew(gamma, axis=0)
                kurtosis_label = kurtosis(gamma, axis=0)
                median_label = np.median(gamma, axis=0)
                iqr_label = np.percentile(gamma, 75, axis=0) - np.percentile(gamma, 25, axis=0)
                range_label = np.max(gamma, axis=0) - np.min(gamma, axis=0)
                
               
                
                # Update the results based on the group
                results[subj_idx-1][0+num_measures*2][ii] = results[subj_idx-1][0+num_measures*2][ii]+ max_label
                results[subj_idx-1][1+num_measures*2][ii] = results[subj_idx-1][1+num_measures*2][ii]+ mean_label
                results[subj_idx-1][2+num_measures*2][ii] = results[subj_idx-1][2+num_measures*2][ii] + median_label
                results[subj_idx-1][3+num_measures*2][ii] = results[subj_idx-1][3+num_measures*2][ii] + iqr_label#!!!
                results[subj_idx-1][4+num_measures*2][ii] = results[subj_idx-1][4+num_measures*2][ii] + range_label
                results[subj_idx-1][5+num_measures*2][ii] = results[subj_idx-1][5+num_measures*2][ii] + percentiles[0]  # 25th percentile #!
                results[subj_idx-1][6+num_measures*2][ii] = results[subj_idx-1][6+num_measures*2][ii] + percentiles[1]  # 75th percentile #!!
                results[subj_idx-1][7+num_measures*2][ii] = results[subj_idx-1][7+num_measures*2][ii] + cv_label
                results[subj_idx-1][8+num_measures*2][ii] = results[subj_idx-1][8+num_measures*2][ii] + signal_energy_label
                results[subj_idx-1][9+num_measures*2][ii] = results[subj_idx-1][9+num_measures*2][ii] + entropy_label
                results[subj_idx-1][10+num_measures*2][ii] = results[subj_idx-1][10+num_measures*2][ii] + kurtosis_label #?
                results[subj_idx-1][11+num_measures*2][ii] = results[subj_idx-1][11+num_measures*2][ii] + variance_label
                results[subj_idx-1][12+num_measures*2][ii] = results[subj_idx-1][12+num_measures*2][ii] + std_deviation_label#!
                results[subj_idx-1][13+num_measures*2][ii] = results[subj_idx-1][13+num_measures*2][ii] + min_value_label
                results[subj_idx-1][14+num_measures*2][ii] = results[subj_idx-1][14+num_measures*2][ii] + skewness_label
                results[subj_idx-1][15+num_measures*2][ii] = results[subj_idx-1][15+num_measures*2][ii] + autocorrelation_mean
                results[subj_idx-1][16+num_measures*2][ii] = results[subj_idx-1][16+num_measures*2][ii] + autocorrelation_zero_lag #!
                results[subj_idx-1][17+num_measures*2][ii] = results[subj_idx-1][17+num_measures*2][ii] + autocorrelation_std #!
                results[subj_idx-1][18+num_measures*2][ii] = results[subj_idx-1][18+num_measures*2][ii] + autocorrelation_max
    
                psd = calculate_psd_relative_power(pca_anat)
                results[subj_idx-1][3*num_measures][ii] = results[subj_idx-1][3*num_measures][ii] + psd['delta_power']
                results[subj_idx-1][3*num_measures+1][ii] = results[subj_idx-1][3*num_measures+1][ii] + psd['theta_power']
                results[subj_idx-1][3*num_measures+2][ii] = results[subj_idx-1][3*num_measures+2][ii] + psd['alpha_power']
                results[subj_idx-1][3*num_measures+3][ii] = results[subj_idx-1][3*num_measures+3][ii] + psd['beta_power']#!!!
                results[subj_idx-1][3*num_measures+4][ii] = results[subj_idx-1][3*num_measures+4][ii] + psd['gamma_power']
                results[subj_idx-1][3*num_measures+5][ii] = results[subj_idx-1][3*num_measures+5][ii] + psd['delta_theta_ratio']
                results[subj_idx-1][3*num_measures+6][ii] = results[subj_idx-1][3*num_measures+6][ii] + psd['theta_alpha_ratio'] #!!!
            except Exception as exc:
                print(exc)
            ii = ii+1
        # for element in results[subj_idx][0]:
        #     element = element/num_segments
        # for element in results[subj_idx][1]:
        #     element = element/num_segments
        # total_segments[avg_idx] = total_segments[avg_idx] + 1
        break
        print(f"done with {subj_idx}")


i = 0
ii = 0
iii = 0

results_for_plot = [[[] for _ in range(total_measures*3)] for _ in labels]
i = 0

for subj in range(len(results)-1):
    ii = 0
    for label in labels:
        if subj_groups[i]=='C':
            for iii in range(total_measures):
                results_for_plot[ii][iii].append(results[i][iii][ii])
        if subj_groups[i]=='F':
            for iii in range(total_measures):
                results_for_plot[ii][iii+total_measures].append(results[i][iii][ii])
        if subj_groups[i]=='A':
            for iii in range(total_measures):
                results_for_plot[ii][iii+total_measures*2].append(results[i][iii][ii])
        ii = ii +1
    i = i + 1    
    
for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
                          locals().items())), key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))   
# label_encoder = sklearn.preprocessing.LabelEncoder()
# train_labels = label_encoder.fit_transform(train_labels)
# val_labels = label_encoder.transform(val_labels)
# test_labels = label_encoder.transform(test_labels)

# # Convert train_data and val_data to tensors of the same shape
# train_data = tf.convert_to_tensor(train_data, dtype=tf.float32)
# val_data = tf.convert_to_tensor(val_data, dtype=tf.float32)
# test_data = tf.convert_to_tensor(test_data, dtype=tf.float32)

# # Convert train_labels and val_labels to tensors
# train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int32)
# val_labels = tf.convert_to_tensor(val_labels, dtype=tf.int32)
# test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int32)


    

subj_groups = []
for subj_idx in subject_indices:
    # Generate subject IDs in the format "001" to "088"
    subject = f'00{subj_idx}' if subj_idx <= 9 else f'0{subj_idx}'
    print(subj_idx)
    # Define the BIDS path for the subject's EEG data
    bids_path = BIDSPath(root=bids_root, subject=subject, datatype=datatype)
    bids_path = bids_path.update(task=task, suffix=suffix)

    # Read the raw EEG data for the subject
    raw = mne.io.read_raw_eeglab(bids_path, verbose=False)

    # Get the total duration of the EEG recording
    total_duration = raw.times[-1]

    # Get the group (A, C, or F) and score for the subject from the 'arr' array
    group = arr[subj_idx][3]
    score = arr[subj_idx][4]
    print(group)
    subj_groups.append(group)

def replace_outliers_with_none(input_list, z_score_threshold=3):
    # Convert the list to a NumPy array
    arr = np.array(input_list, dtype=float)  # Convert elements to float for NaN handling

    # Calculate the z-scores for the elements in the array
    z_scores = np.abs((arr - np.nanmean(arr)) / np.nanstd(arr))
    
    # Identify outliers based on the z-score threshold
    outliers = z_scores > z_score_threshold
    
    # Replace outliers with None
    arr[outliers] = np.mean(arr)

    # Convert the NumPy array back to a list
    output_list = arr.tolist()
    
    return output_list


i = 0
results_for_plot_no_outliers = [[[] for _ in range(total_measures*3)] for _ in labels]
for ii in range(total_measures):
    for label in labels:
        results_for_plot_no_outliers[i][ii] = replace_outliers_with_none(results_for_plot[i][ii])
        results_for_plot_no_outliers[i][ii+total_measures] = replace_outliers_with_none(results_for_plot[i][ii+total_measures])
        results_for_plot_no_outliers[i][ii+total_measures*2] = replace_outliers_with_none(results_for_plot[i][ii+total_measures*2])
        i = i + 1
    i = 0

# i = 0
# maxdiff = [0, 0, 0]
# minmax = np.empty((192,2))
# for ii in range(total_measures):
#     for label in labels:
        
        
#             # if(abs(np.mean(results_for_plot[i][ii+total_measures*2]))>0):
#             #     if(maxdiff[0]<(abs(np.mean(results_for_plot[i][ii+total_measures*2])-np.mean(results_for_plot[i][ii]))/max(abs(results_for_plot[i][ii+total_measures*2])),abs(np.mean(results_for_plot[i][ii])))):
#             #         maxdiff[0] =  abs(np.mean(results_for_plot[i][ii+total_measures*2])-np.mean(results_for_plot[i][ii]))/max(abs(results_for_plot[i][ii+total_measures*2])),abs(np.mean(results_for_plot[i][ii]))
#             #         maxdiff[1] = i
#             #         maxdiff[2] = ii
#         if(min(results_for_plot_no_outliers[i][ii])<minmax[ii][0]):
#             minmax[ii][0]=min(results_for_plot_no_outliers[i][ii])
#         if(max(results_for_plot_no_outliers[i][ii])>minmax[ii][1]):
#             minmax[ii][1]=max(results_for_plot_no_outliers[i][ii])
#         if(min(results_for_plot_no_outliers[i][ii+total_measures])<minmax[ii+total_measures][0]):
#             minmax[ii+total_measures][0]=min(results_for_plot_no_outliers[i][ii+total_measures])
#         if(max(results_for_plot_no_outliers[i][ii+total_measures])>minmax[ii+total_measures][1]):
#             minmax[ii+total_measures][1]=max(results_for_plot_no_outliers[i][ii+total_measures])
#         if(min(results_for_plot_no_outliers[i][ii+total_measures*2])<minmax[ii+total_measures*2][0]):
#             minmax[ii+total_measures*2][0]=min(results_for_plot_no_outliers[i][ii+total_measures*2])
#         if(max(results_for_plot_no_outliers[i][ii+total_measures*2])>minmax[ii+total_measures*2][1]):
#             minmax[ii+total_measures*2][1]=max(results_for_plot_no_outliers[i][ii+total_measures*2])
       
#         i = i + 1
#     i = 0


def normalize_and_floor_list(input_list1, input_list2, input_list3, new_max=1):
    min_val = min(min(input_list1), min(input_list2), min(input_list3))
    max_val = max(max(input_list1), max(input_list2), max(input_list3))
    range_val = max_val - min_val
    if range_val == 0:
        # Handle the case where all values are the same
        return [int(new_max)] * len(input_list1), [int(new_max)] * len(input_list2), [int(new_max)] * len(input_list3)
    normalized_list1 = []
    for x in input_list1:
        try:
            normalized_value = int((x - min_val) / range_val * new_max)
            print(normalized_value)
            normalized_list1.append(normalized_value)
        except:
            normalized_list1.append(x)
            
    normalized_list2 = []
    for x in input_list2:
        try:
            normalized_value = int((x - min_val) / range_val * new_max)
            print(normalized_value)
            normalized_list2.append(normalized_value)
        except:
            normalized_list2.append(x)
    
    normalized_list3 = []
    for x in input_list3:
        try:
            normalized_value = int((x - min_val) / range_val * new_max)
            print(normalized_value)
            normalized_list3.append(normalized_value)
        except:
            normalized_list3.append(x)
    return normalized_list1, normalized_list2, normalized_list3

def normalize_and_floor_list_subj(input_list, new_max=255**3):
    min_val = min(input_list)
    max_val = max(input_list)
    range_val = max_val - min_val
    if range_val == 0:
        # Handle the case where all values are the same
        return [int(new_max)] * len(input_list)
    normalized_list = []
    for x in input_list:
        try:
            normalized_value = int((x - min_val) / range_val * new_max)
            print(normalized_value)
            normalized_list.append(normalized_value)
        except:
            normalized_list.append(x)
            
    return normalized_list


results_norm = np.zeros((num_subjects, total_measures, len(labels)))
for label_index, label in enumerate(labels):
    for measure in range(total_measures):
        temp_list = [result[measure][label_index] for result in results]
        list_no_outliers = replace_outliers_with_none(temp_list)
        list_norm = normalize_and_floor_list_subj(list_no_outliers, (255**3))
        for norm_idx, norm in enumerate(list_norm):
            results_norm[norm_idx][measure][label_index] = norm

i = 0
max_val=255**3
results_for_plot_normalized = [[[] for _ in range(total_measures*3)] for _ in labels]
for ii in range(total_measures):
    for label in labels:
        results_list = normalize_and_floor_list(results_for_plot_no_outliers[i][ii],results_for_plot_no_outliers[i][ii+total_measures], results_for_plot_no_outliers[i][ii+total_measures*2], max_val)
        results_for_plot_normalized[i][ii] = results_list[0]
        results_for_plot_normalized[i][ii+total_measures] = results_list[1]
        results_for_plot_normalized[i][ii+total_measures*2] = results_list[2]
        i = i + 1
    i = 0


# i = 0
# for ii in range(int(total_measures/10)):
#     for label in labels:
#         try:
#             plt.hist(results_for_plot_normalized[i][ii], bins=50, density=True, alpha=0.6, color='g')
#             #plt.hist(results_for_plot_normalized[i][ii+total_measures], bins=50, density=True, alpha=0.6, color='b')
#             plt.hist(results_for_plot_normalized[i][ii+total_measures*2], bins=50, density=True, alpha=0.6, color='r')
#             plt.xlabel('Value')
#             plt.ylabel('Probability density')
#             plt.title(f"{label.name} {measures[ii]}")
#             plt.show()
#         except:
#             print("could not plot")
#         i = i + 1
#     i = 0
    
    
from sklearn.model_selection import train_test_split

results_norm_arr = np.nan_to_num(np.array(results_norm))
results_norm_arr_int = results_norm_arr.astype(int)
results_AC = []
groups_AC = []
for subj_index, subject in enumerate(results_norm_arr[:-1]):
    if subj_groups[subj_index] == "A" or subj_groups[subj_index] == "C":
        results_AC.append(subject)
        if subj_groups[subj_index] == "A":
            groups_AC.append(1)
        if subj_groups[subj_index] == "C":
            groups_AC.append(0)   
        
data = np.array(results_AC)
labels = np.array(groups_AC)
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# from PIL import Image
# for idmatrix, matrix in enumerate(results_norm_arr_int):
#     image = Image.fromarray(matrix, mode="RGB")
#     image = image.resize((69*3, 64*3))
#     image.save(f"matrix/{subj_groups[idmatrix]}_{idmatrix}.png")


# shapes = set(matrix.shape for matrix in [results_norm_arr_int[0], results_norm_arr_int[0]])
# if len(shapes) != 1:
#     raise ValueError("Matrices must have the same shape")

# # Sum the matrices element-wise
# summed_matrix = np.sum([matrix for matrix in [results_norm_arr_int[0], results_norm_arr_int[0]]], axis=0)

# # Divide the summed matrix by the number of matrices to get the average
# num_matrices = len([matrix for matrix in [results_norm_arr_int[0], results_norm_arr_int[0]]])
# average_matrix = summed_matrix / num_matrices

# image = Image.fromarray(average_matrix, mode="RGB")
# image = image.resize((69*3, 64*3))
# image.save(f"matrix/C_avg.png")

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

from tabpfn import TabPFNClassifier
import time
from sklearn.metrics import accuracy_score
import warnings
import gc
from sklearn.ensemble import RandomForestClassifier
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns

def remove_models_from_memory():
    # Iterate over all objects in the global scope
    for obj_name in list(globals()):
        obj = globals()[obj_name]
        if isinstance(obj, torch.nn.Module):
            # If the object is a PyTorch model, delete it
            del globals()[obj_name]


warnings.filterwarnings("ignore", category=UserWarning)

features_flat = []
for arr in data:
    features_flat.append(arr.flatten())
gc.collect()

k_best = SelectKBest(chi2, k=10)  # Select the top 2 features

# Fit the SelectKBest instance to the data and transform the data
X_new = k_best.fit_transform(features_flat, labels)

success_arr = []
predictions = []
predictions_prob = []
for current in range(len(X_new)):
    remove_models_from_memory()
    print(f"Leave-One-Subject-Out: Testing on subject {current + 1}/{len(X_new)}")
    classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=5)
    x_train = X_new.tolist()
    y_train = labels.tolist()
    x_test = np.array(x_train.pop(current)).reshape(1,-1)
    x_train = np.array(x_train)
    y_test = np.array(y_train.pop(current))
    y_train = np.array(y_train)
    classifier.fit(x_train, y_train)
    y_pred, p_pred = classifier.predict(x_test, return_winning_probability=True)
    #clf = RandomForestClassifier(n_estimators=1000, random_state=42)
    
    # Fit the classifier to the training data]
    #clf.fit(x_train, y_train)
    
    # Make predictions on the test data
    #y_pred = clf.predict(x_test)
    predictions.append(y_pred)
    predictions_prob.append(p_pred)
    
confusion_mtx = confusion_matrix(labels, predictions)

# Define class names
class_names = ["Control group(0)", "Alzheimer group(1)"]

# Calculate accuracy, sensitivity, specificity, precision, and F1-score
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)
specificity = confusion_mtx[0, 0] / (confusion_mtx[0, 0] + confusion_mtx[0, 1])

# Output metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Sensitivity (Recall):", recall)
print("Specificity:", specificity)
print("F1-score:", f1)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

fpr, tpr, thresholds = roc_curve(labels, predictions_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
