# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.scimath import sqrt
import tidy3d as td 
from tidy3d import web
import os
from scipy.optimize import curve_fit
from scipy import optimize
from scipy.interpolate import CubicSpline
import csv
import pandas as pd
import trimesh
from IPython.display import Image, display

#for now there is a last sample layer of nk_sample, but there is also a semi-infinite sample layer nk_sample, these can be changed

def TMM_setup_layers(nk_prism, nk_layers, nk_sample, nk_backlayer, sample_thickness,  layer_thicknesses, period, incident_angle, wavelength):
    # Calculates reflectivity and phase by TMM methods
    # n1 = prism refractive index
    # n2 = gold refractive index
    # n3 = sample refractive index
    # incident_angle = incident angle from substance 1 onto prism (From normal)
    # tg = gold thickness
    # ts = sample thickness per layer
    # ts_total = total sample thickness
    # layering = 'on' or 'off' to use multiple layers for thick samples

    thicknesses = layer_thicknesses
    layers = len(nk_layers)
    #PRISM TO material """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    E1 = nk_prism**2
    a = (nk_prism*np.sin(incident_angle))**2

    Q_prism_p =  sqrt(E1 - a) / E1
    Q_prism_s =  Q_prism_p * E1    
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    #Add the layer matricies$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #setting up list containg matricies for each material
    layers_s= []
    layers_p=[]
    
    for i in range(layers):
        #define n_k data and thickness at each layer 
        nk = nk_layers[i]
        E = (nk_layers[i])**2
        t = thicknesses[i]

        B = ((2 * np.pi * t) / wavelength) * sqrt(E - a)
        
        Q_p = sqrt(E - a) / E
        Q_s = Q_p * E
        
        p2m_M11 = np.cos(B)
        p2m_M12p = (-1j * np.sin(B)) / Q_p
        p2m_M21p = -1j * Q_p * np.sin(B)
        p2m_M22 = np.cos(B)
        
        p2m_M12s = (-1j * np.sin(B)) / Q_s
        p2m_M21s = -1j * Q_s * np.sin(B)
        
        
        #PRISM TO GOLD
        layer_matrix_p = np.array([[p2m_M11, p2m_M12p],
                              [p2m_M21p, p2m_M22]])
        
        layer_matrix_s = np.array([[p2m_M11, p2m_M12s],
                              [p2m_M21s, p2m_M22]])

        layers_p.append(layer_matrix_p)
        layers_s.append(layer_matrix_s)
        #print(layers_p)
        #remember the layers must be dotted from the sample up so reverse the list order

    layers_p = layers_p[::-1]
    layers_s = layers_s[::-1]

    if layers > 1:
        #create total matrix for mid section (not prism or sample)
        layers_p = np.linalg.multi_dot(layers_p)
        layers_s = np.linalg.multi_dot(layers_s)
    
        #choose how many times you want the layered section to repeat 
        layers_p = np.linalg.matrix_power(layers_p, period)
        layers_s = np.linalg.matrix_power(layers_s, period)

    else:
        print("Just one layer wow not very cool")
        print(layers_p, layers_s)
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
    #SAMPLE LAYER now add the sample layer at the bottom###############################################################################
    
    #define n_k data and thickness at each layer 
    E2 = nk_sample**2
    t2 = sample_thickness

    B2 = ((2 * np.pi * t2) / wavelength) * sqrt(E2 - a)
    
    Q2_p = sqrt(E2 - a) / E2
    Q2_s = Q2_p * E2
    
    sample_M11 = np.cos(B2)
    sample_M12p = (-1j * np.sin(B2)) / Q2_p
    sample_M21p = -1j * Q2_p * np.sin(B2)
    sample_M22 = np.cos(B2)
    
    sample_M12s = (-1j * np.sin(B2)) / Q2_s
    sample_M21s = -1j * Q2_s * np.sin(B2)
    
    
    #PRISM TO GOLD
    sample_matrix_p = np.array([[sample_M11, sample_M12p],
                          [sample_M21p, sample_M22]])
    
    sample_matrix_s = np.array([[sample_M11, sample_M12s],
                          [sample_M21s, sample_M22]])

    ################################################################################################################################
    
    # Multiply matrices
    TMM_matrix_p = np.dot(sample_matrix_p, layers_p)
    TMM_matrix_s = np.dot(sample_matrix_s, layers_s)
    
    Mp = TMM_matrix_p
    M11p = Mp[0,0]
    M12p = Mp[0,1]
    M21p = Mp[1,0]
    M22p = Mp[1,1]


    Ms = TMM_matrix_s
    M11s = Ms[0,0]
    M12s = Ms[0,1]
    M21s = Ms[1,0]
    M22s = Ms[1,1]

    #define backlayer
    Eb = nk_backlayer**2
    Qb_p = sqrt(Eb - a) / Eb
    Qb_s = Qb_p * Eb

    # Calculate fresnel coefficients must use last layer eg sample q values
    qN_p = Qb_p

    #below use first layer
    q1_p = Q_prism_p 

    r_p = ((M11p + M12p * qN_p) * q1_p - (M21p + M22p * qN_p)) / \
          ((M11p + M12p * qN_p) * q1_p + (M21p + M22p * qN_p))
    p_phase = np.angle(r_p)
    p_reflectivity = (np.abs(r_p))**2

    #last layer q
    qN_s = Qb_s

    #first layer
    q1_s = Q_prism_s

    r_s = ((M11s + M12s * qN_s) * q1_s - (M21s + M22s * qN_s)) / \
          ((M11s + M12s * qN_s) * q1_s + (M21s + M22s * qN_s))
    s_phase = np.angle(r_s)
    s_reflectivity = (np.abs(r_s))**2

    return s_phase, p_phase, s_reflectivity, p_reflectivity






def scan_angle(start_angle,end_angle, nk_prism, nk_layers, nk_sample,nk_backlayer,sample_thickness,  layer_thicknesses, wavelength, period,
               plot,scans):
    #set up angles list
    angles = np.linspace(start_angle, end_angle, scans)
    layers = len(nk_layers)
    #lists for s and p polarisation phase and reflection
    sphas = []
    pphas = []
    sref = []
    pref = []

    for angle in angles:
        # Convert degrees to radians
        incident_angle = np.radians(angle)

        # Call your TMM_setup function here
        spha, ppha, sre, pre = TMM_setup_layers(nk_prism=nk_prism, nk_layers=nk_layers, nk_sample=nk_sample,sample_thickness=sample_thickness,  layer_thicknesses=layer_thicknesses, incident_angle=incident_angle, wavelength=wavelength,period=period, nk_backlayer=nk_backlayer)
        
        sphas.append(spha)
        pphas.append(ppha)
        sref.append(sre)
        pref.append(pre)

    #make lists into arrays for numpy operations
    sphas = np.array(sphas)
    pphas = np.array(pphas)
    sref = np.array(sref)
    pref = np.array(pref)

    # Unwrap the phase in degrees
    unw_pphas = np.degrees(np.unwrap(pphas))
    unw_sphas = np.degrees(np.unwrap(sphas))

    # Calculate Goos-Hänchen shifts
    gh_shift_p = ((-1 * wavelength) / (2 * np.pi)) * np.gradient(unw_pphas, angles)
    gh_shift_s = ((-1 * wavelength) / (2 * np.pi)) * np.gradient(unw_sphas, angles)
    diff_gh = gh_shift_p - gh_shift_s

    # Full 3x3 plot (existing plotting mode)
    if plot == 'on':
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))  # Create a 3x3 grid of subplots

        # Wrapped S phase
        axes[0, 0].plot(angles, sphas)
        axes[0, 0].set_title("Wrapped S phase")

        # Unwrapped S phase
        axes[0, 1].plot(angles, unw_sphas)
        axes[0, 1].set_title("Unwrapped S phase")

        # Lateral Goos-Hänchen shift of S polarization
        axes[0, 2].plot(angles, gh_shift_s)
        axes[0, 2].set_title("Lateral Goos-Hänchen shift (S polarization)")

        # Wrapped P phase
        axes[1, 0].plot(angles, pphas)
        axes[1, 0].set_title("Wrapped P phase")

        # Unwrapped P phase
        axes[1, 1].plot(angles, unw_pphas)
        axes[1, 1].set_title("Unwrapped P phase")

        # Lateral Goos-Hänchen shift of P polarization
        axes[1, 2].plot(angles, gh_shift_p)
        axes[1, 2].set_title("Lateral Goos-Hänchen shift (P polarization)")

        # Differential Goos-Hänchen shift
        axes[2, 0].plot(angles, diff_gh)
        axes[2, 0].set_title("Differential Goos-Hänchen shift")

        # P reflectivity
        axes[2, 1].plot(angles, pref, label='P reflectivity')
        axes[2, 1].plot(angles, sref, label='S reflectivity')
        axes[2, 1].legend()
        axes[2, 1].set_title("Reflectivity")

        # S reflectivity
        axes[2, 2].plot(angles, sref)
        axes[2, 2].set_title("S reflectivity")

        plt.tight_layout()
        plt.show()
        #print(np.min(diff_gh))

    # Simple 1x2 plot with reflectivity and differential GH shift
    if plot == 'simple':
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Reflectivity (both P and S)
        axes[0].plot(angles, pref, label='P reflectivity')
        axes[0].plot(angles, sref, label='S reflectivity')
        axes[0].set_xlabel("Angle (degrees)")
        axes[0].set_ylabel("Reflectivity")
        axes[0].set_title("Reflectivity")
        axes[0].legend()

        # Differential GH shift
        axes[1].plot(angles, diff_gh, label='GH shift (P-S)')
        axes[1].set_xlabel("Angle (degrees)")
        axes[1].set_ylabel("GH shift (metres)")
        axes[1].set_title("Differential Goos-Hänchen shift")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return unw_pphas, unw_sphas, sphas, pphas, sref, pref, gh_shift_p, gh_shift_s, diff_gh, angles


def layer_sweep(start_angle,end_angle, nk_prism, nk_layers, nk_sample,nk_backlayer,sample_thickness,layer_list, wavelength, period, plot,scans):
    # Lists that will hold scan_angle values
    graphs = plot
    sphas_array = []
    pphas_array = []
    unw_pphas_array = []
    unw_sphas_array = []
    sref_array = []
    pref_array = []
    gh_shift_p_array = []
    gh_shift_s_array = []
    diff_gh_array = []
    labels = []

    max_diff_gh_values=[]

    for i, layer_val in enumerate(layer_list):

        (
        unw_pphas, 
        unw_sphas, 
        sphas, 
        pphas, 
        sref, 
        pref, 
        gh_shift_p, 
        gh_shift_s, 
        diff_gh, 
        angles
    ) = scan_angle(start_angle=start_angle,
                   end_angle=end_angle,
                   nk_prism=nk_prism, 
                   nk_layers=nk_layers, 
                   nk_sample=nk_sample,
                   nk_backlayer=nk_backlayer,
                   sample_thickness=sample_thickness,  
                   layer_thicknesses=layer_val, 
                   wavelength=wavelength, 
                   period=period,
                   plot='off',
                   scans=scans)


        # Collect the results
        unw_pphas_array.append(unw_pphas)
        unw_sphas_array.append(unw_sphas)
        sphas_array.append(sphas)
        pphas_array.append(pphas)
        sref_array.append(sref)
        pref_array.append(pref)
        gh_shift_p_array.append(gh_shift_p)
        gh_shift_s_array.append(gh_shift_s)
        diff_gh_array.append(diff_gh)

        # Make a label for plotting
        labels.append(f"layer_val = {layer_val}")
    
        # Compute the maximum absolute differential GH shift for this specific metal index
        max_diff_gh = np.max(np.abs(diff_gh))
        max_diff_gh_values.append(max_diff_gh)
        #print(max_diff_gh_values)
        #print(index_positions)
    # ------------------
    #       PLOTTING
    # ------------------
    if graphs == 'complex':
        # 9 total plots: wrapped S/P phases, unwrapped S/P phases, GH shifts (S, P, diff), S & P reflectivities
        num_plots = 9
        num_cols = 3
        num_rows = int(np.ceil(num_plots / num_cols))

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 4))
        axs = axs.flatten()

        plot_data = [
            (sphas_array,      'Wrapped S Phase vs Angle', 'Wrapped S Phase'),
            (pphas_array,      'Wrapped P Phase vs Angle', 'Wrapped P Phase'),
            (unw_sphas_array,  'Unwrapped S Phase vs Angle', 'Unwrapped S Phase'),
            (unw_pphas_array,  'Unwrapped P Phase vs Angle', 'Unwrapped P Phase'),
            (gh_shift_p_array, 'GH Shift (P) vs Angle', 'GH Shift P'),
            (gh_shift_s_array, 'GH Shift (S) vs Angle', 'GH Shift S'),
            (diff_gh_array,    'Differential GH Shift vs Angle', 'GH Shift (P-S)'),
            (sref_array,       'S Reflectivity vs Angle', 'Reflectivity'),
            (pref_array,       'P Reflectivity vs Angle', 'Reflectivity'),
        ]

        for ax_idx, (data_array, title, ylabel) in enumerate(plot_data):
            for data, label in zip(data_array, labels):
                axs[ax_idx].plot(angles, data, label=label)
            axs[ax_idx].set_title(title)
            axs[ax_idx].set_xlabel('Angle (degrees)')
            axs[ax_idx].set_ylabel(ylabel)
            axs[ax_idx].grid(True)
            axs[ax_idx].legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Hide any unused subplots (in case num_plots < len(axs))
        for idx in range(len(plot_data), len(axs)):
            fig.delaxes(axs[idx])

        plt.tight_layout()
        plt.show()

    elif graphs == 'simple':
        # Only plot GH shift (P) and P reflectivity
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # GH Shift (P)
        for data, label in zip(gh_shift_p_array, labels):
            axs[0].plot(angles, data, label=label)
        axs[0].set_title('GH Shift (P) vs Angle')
        axs[0].set_xlabel('Angle (degrees)')
        axs[0].set_ylabel('GH Shift P')
        axs[0].grid(True)
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

        # P Reflectivity
        for data, label in zip(pref_array, labels):
            axs[1].plot(angles, data, label=label)
        axs[1].set_title('P Reflectivity vs Angle')
        axs[1].set_xlabel('Angle (degrees)')
        axs[1].set_ylabel('P Reflectivity')
        axs[1].grid(True)
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()

    elif graphs == 'shift':
        # Plot maximum differential GH shift vs index *position* in the list
        # (We could also parse real/imag parts, but that might be less straightforward.)
        plt.figure(figsize=(8, 6))
        index_positions = np.arange(len(layer_list))  # 0, 1, 2, ...
        plt.plot(index_positions, max_diff_gh_values, marker='o')
        plt.title('Maximum Differential GH Shift vs Metal Index (List Position)')
        plt.xlabel('Index in layer_list')
        plt.ylabel('Maximum Differential GH Shift')
        plt.grid(True)
        plt.legend(['Max Diff GH Shift'], loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()
    index_positions = np.arange(len(layer_list))
    # ---------------
    #  RETURN VALUE
    # ---------------
    # Also compute GH shift at resonance and track the maximum absolute GH shift
    gh_sens = []
    gh_peaks = []

    for i in range(len(layer_list)):
        # Resonance index is where P-reflectivity is minimum
        res_ind = np.argmin(pref_array[i])
        gh_at_spr = diff_gh_array[i][res_ind]
        gh_sens.append(gh_at_spr)

        # The maximum absolute GH shift across all angles for this particular layer_val
        gh_peak_ind = np.argmax(np.abs(diff_gh_array[i]))
        gh_peak_val = diff_gh_array[i][gh_peak_ind]
        gh_peaks.append(gh_peak_val)

    # The global maximum
    max_gh_shift = np.max(np.abs(gh_peaks))
    max_ind = np.argmax(max_diff_gh_values)

    return max_gh_shift, max_ind, max_diff_gh_values, index_positions









