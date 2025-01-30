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


def set_nano_sim(
    wavelength_range,
    Nfreqs,
    spacing,#spacing between holes
    metal_thickness,#metal film thickness
    di_thickness, #dielectric filkm thickness
    metal_material,#using JC gold for now
    dielectric_material, #hole material
    prism_material,
    sample_material,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    sim_height,
    sim_centre,
    ):
    
    #SET PARAMATERS 
    
    #n_prism = np.sqrt(prism_material)
    stack_point =0
    sp = stack_point

    meta_thick = 2*metal_thickness + di_thickness
    
    min_wavelength, max_wavelength = wavelength_range
    
    #frequency and wavelength paramaters
    
    central_wavelength =np.mean(wavelength_range)
    lda0 = central_wavelength  # central wavelength
    freq0 = td.C_0 / lda0  # central frequency
    ldas = np.linspace(min_wavelength, max_wavelength, Nfreqs)  # wavelength range
    freqs = td.C_0 / ldas  # frequency range
    fwidth = 0.5 * (np.max(freqs) - np.min(freqs))  # width of the source frequency range


    
    #define geometry for nanohole array
    a = spacing #cylinder spacing

    material = metal_material
    #CREATE UP TO FIVE HOLES FOR A UNIT CELL
    

    # define the metal base plate structure
    metal_base2 = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf,sp),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, metal_thickness)          # Upper bound: Top of the base at z = 0
    ),
    medium=metal_material,
    name="MetalBase2"
    )
    
    # define the metal base plate structure
    metal_base = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, metal_thickness+di_thickness),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, metal_thickness+di_thickness+metal_thickness)          # Upper bound: Top of the base at z = 0
    ),
    medium=metal_material,
    name="MetalBase"
    )

    # define the dielectric base plate structure
    di_base = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, metal_thickness),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, metal_thickness+di_thickness)          # Upper bound: Top of the base at z = 0
    ),
    medium=dielectric_material,
    name="DielectricBase"
    )

    
    MetalMesh = td.MeshOverrideStructure(
        geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, metal_thickness+di_thickness),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf,  metal_thickness+di_thickness+metal_thickness)          # Upper bound: Top of the base at z = 0
    ),
        dl=(metal_thickness/mesh_res, metal_thickness/mesh_res ,metal_thickness/mesh_res),
        name ='MetalMesh'
)


    MetalMesh2 = td.MeshOverrideStructure(
        geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, sp),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf,  metal_thickness)          # Upper bound: Top of the base at z = 0
    ),
        dl=(metal_thickness/mesh_res, metal_thickness/mesh_res ,metal_thickness/mesh_res),
        name ='MetalMesh2'
)

    DiMesh = td.MeshOverrideStructure(
        geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, sp),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf,  metal_thickness)          # Upper bound: Top of the base at z = 0
    ),
        dl=(di_thickness/mesh_res, di_thickness/mesh_res ,di_thickness/mesh_res),
        name ='DiMesh'
)

    prism = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, metal_thickness+di_thickness+metal_thickness),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, 100000)          # Upper bound: Top of the base at z = 0
    ),
    medium=prism_material,
    name="Prism"
    )

    sample = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, -10000000),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, sp)          # Upper bound: Top of the base at z = 0
    ),
    medium=sample_material,
    name="Sample"
    )

    '''    # Combine all holes into a single group
    hole_cell = [hole_1, hole_2, hole_3, hole_4, hole_5]
    holes = td.GeometryGroup(hole_cell)'''
    
    

    #now define the flux monitors
    flux_monitor = td.FluxMonitor(
    center=[0, 0, central_wavelength*1.2+meta_thick], size=[td.inf, td.inf, 0], freqs=freqs, name="R"
    )
    '''
    flux_monitor2 = td.FluxMonitor(
    center=[0, 0, -1.2*central_wavelength ], size=[td.inf, td.inf, 0], freqs=freqs, name="T"
    )
    '''
    '''
    monitor_field = td.FieldMonitor(
    center=[0,0,0],
    size= [0, np.inf,np.inf],
    freqs=freq0,
    name="field",
    )
    
    monitor_field2 = td.FieldMonitor(
    center=[d/2,0,0],
    size= [0, 0,np.inf],
    freqs=freq0,
    name="field2",
    )
    '''
    #point monitor for field phase
    r_phase_monitor = td.FieldMonitor(
    center=[0,0,central_wavelength*1.2+meta_thick],
    size= [0, 0,0],
    freqs=freq0,
    name="phase",
    )
    '''
    side = td.FieldMonitor(
    center=[0,0,sp+meta_thick/2],
    size= [0, np.inf,meta_thick],
    freqs=freq0,
    name="side",
    )
    top = td.FieldMonitor(
    center=[0,0,sp+h],
    size= [np.inf,np.inf,0],
    freqs=freq0,
    name="top",
    )
    bottom = td.FieldMonitor(
    center=[0,0,sp],
    size= [np.inf,np.inf,0],
    freqs=freq0,
    name="bottom",
    )
    '''
    
    
    def make_sim(angle):
        
        
        
        #define source
        plane_wave = td.PlaneWave(
        source_time= td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        size=(a, a, 0),
        center=(0, 0, central_wavelength+meta_thick),
        direction="+",
        pol_angle=pol_angle,
        angle_phi = np.pi/2,
        angle_theta = np.radians(180-angle),
        )
        
        run_time = run_t / freq0  # simulation run time

        # define simulation domain size
        sim_size = (a,  a,  sim_height*central_wavelength)
        # create the Bloch boundaries
        bloch_x = td.Boundary.bloch_from_source(
        source=plane_wave, domain_size=sim_size[0], axis=0, medium=prism_material
        )
        bloch_y = td.Boundary.bloch_from_source(
        source=plane_wave, domain_size=sim_size[1], axis=1, medium=prism_material
        )

        bspec_bloch = td.BoundarySpec(x=bloch_x, y=bloch_y, z=td.Boundary.pml())
        # Calculate wavevector components and Bloch wavevectors


        grid_spec = td.GridSpec.auto(
        min_steps_per_wvl = grid_res,
        override_structures = [MetalMesh, MetalMesh2,DiMesh],
        wavelength=lda0,
        )
        
        
            
        structures= [sample, metal_base, di_base, prism, metal_base2] 


        sim = td.Simulation(
        center=(0, 0, sim_centre),
        size=sim_size,
        grid_spec=grid_spec,
        structures = structures,
        sources=[plane_wave],
        monitors=[flux_monitor, r_phase_monitor],#,monitor_field monitor_field2, side, bottom, top],
        run_time=run_time,
        boundary_spec=bspec_bloch, 
        #symmetry=(1, -1, 0),
        #shutoff=1e-7,  # reducing the default shutoff level
        )
    
        return sim

        #ax = sim.plot(z=h / 2)
        #sim.plot_grid(z=h / 2, ax=ax)

    hole_bot = 0
    hole_side = 0
        
        
    
    return make_sim, hole_bot, hole_side, freq0,

def reflectivity( r_data, freq0):


    #get transmitted flux and evaluate value at only central frequnecy
    ref_flux = r_data.flux 
    r = np.abs(ref_flux.sel(f=freq0))
    
    reflectance = (r).item()
    return reflectance

def ref_multi(t_data, r_data, wavelength_range, Nfreqs):
    wr = wavelength_range
    min_wavelength, max_wavelength = wavelength_range
    #frequency and wavelength paramaters
    central_wavelength =np.mean(wavelength_range)
    lda0 = central_wavelength  # central wavelength
    freq0 = td.C_0 / lda0  # central frequency
    ldas = np.linspace(min_wavelength, max_wavelength, Nfreqs)  # wavelength range
    freqs = td.C_0 / ldas  # frequency range

    reflectivities=([])
    for i in range(Nfreqs):

        #get transmitted flux and evaluate value at only central frequnecy
        trans_flux = t_data.flux
        t = np.abs(trans_flux.sel(f=freqs[i]))
    
        #the same for reflected
        ref_flux = r_data.flux 
        r = np.abs(ref_flux.sel(f=freqs[i]))
        
        #reflectance 
        i = r + t
        reflectance = (r).item()
        reflectivities.append(reflectance)
        
    return reflectivities
    
def get_phase(phase_data):

    #Ex = phase_data.Ex
    Ey = phase_data.Ey

    #changed to Ex for spol change back lol
    Ez = phase_data.Ex
    E = np.sqrt( Ey**2 + Ez**2)

    phase_y = np.angle(Ey)
    phase_z = np.angle(Ez)
    return phase_y.item(), phase_z.item()

def phase_multi(phase_data, Nfreqs, wavelength_range):

    #Ex = phase_data.Ex
    Ey = phase_data.Ey
    Ez = phase_data.Ez
    E = np.sqrt( Ey**2 + Ez**2)

    #phase_x = np.angle(Ex)
    phase_y = np.angle(Ey)
    phase_z = np.angle(Ez)
    return phase_y.item(), phase_z.item()
    


def nanohole_scan(
    wavelength_range,
    Nfreqs,
    spacing,#spacing between holes
    metal_thickness,#metal film thickness
    di_thickness, #dielectric filkm thickness
    metal_material,#using JC gold for now
    dielectric_material, #hole material
    prism_material,
    sample_material,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    sim_height,
    sim_centre,
    start_angle,
    end_angle,
    steps
    ):
    
    
    make_sim, hole_bot, hole_side, freq0 = set_nano_sim(
    wavelength_range,
    Nfreqs,
    spacing,#spacing between holes
    metal_thickness,#metal film thickness
    di_thickness, #dielectric filkm thickness
    metal_material,#using JC gold for now
    dielectric_material, #hole material
    prism_material,
    sample_material,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    sim_height,
    sim_centre,
    )
    
    central_wavelength=np.mean(wavelength_range)
    angles = np.linspace(start_angle, end_angle ,steps)

        
    #plot one graph to get pricing and plot ---------------------------------------------------------------------------------------
    nano_sim = make_sim(angle=angles[0])
    print("Plotting for angle", angles[0])
    # visualize geometry
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    nano_sim.plot(z=hole_bot, ax=ax1)
    nano_sim.plot(y=hole_side, ax=ax2)

    plt.show()
    
    task_id = web.upload(nano_sim, task_name="Nanohole Array")
    estimated_cost = web.estimate_cost(task_id=task_id)
    print("This simulation will cost", estimated_cost, "flex credits")
    print(steps, "simulations will cost", estimated_cost*steps, "flex credits")
    
    input("Proceed?")
    
    
    angles = np.linspace(start_angle, end_angle, steps)
    p_reflectivities = np.zeros(steps)
    phases = np.zeros(steps)
    phases2 = np.zeros(steps)

    sims = {f"l={l:.2f}": make_sim(l) for l in angles}
    batch = web.Batch(simulations=sims, verbose=True)

    batch_results = batch.run(path_dir="data")
    reflectivity_batch = []
    phase_batch_y = []
    phase_batch_z= []
    

    for task_name, sim_data in batch_results.items():
        trans_data = 5
        ref_data = sim_data['R']
        phase_data = sim_data['phase']
        
        phases_y, phases_z = get_phase(phase_data)

        #calcualtes array of reflectivities at different wavelenths for each angle point
        reflectivities = reflectivity( ref_data, freq0)

        #print(reflectivities)
            
        reflectivity_batch.append(reflectivities)
        phase_batch_y.append(phases_y)
        phase_batch_z.append(phases_z)

    #print(reflectivity_batch)
    ref = reflectivity_batch
    phasez  = phase_batch_z 
    phase = np.array(phasez)
    pz = phase
    
    #unwrap the phase for a smooth curve
    uwpz = np.degrees(np.unwrap(pz))

    #interpolate to make a continous curve
    interp_phase = CubicSpline(angles,(uwpz))

    #differentiate the curve 
    phase_div = interp_phase.derivative(1)(angles)


    #calculate the lateral goos-hanchen shift
    A = -2*np.pi / central_wavelength
    gh_shift = (1/A)*phase_div*1e-6
    
    deg_phase_z = np.degrees(phase_batch_z)
    uphase = np.unwrap(deg_phase_z)
    
    deg_phase_y = np.degrees(phase_batch_y)
    uphase_y = np.unwrap(deg_phase_y)
    
    #plt.figure()
    #plt.title("Reflectivity against angles")
    #plt.plot(angles, reflectivity_batch)
    #plt.show()
    '''
    plt.figure()
    plt.title("Point phase against angles z")
    plt.plot(angles, phase_batch_z)
    plt.show()
    
    plt.figure()
    plt.title("Lateral shift agains angles")
    plt.plot(angles, gh_shift)
    plt.show()

    ref = reflectivity_batch

    wavelength_max = np.max(wavelength_range)
    wavelength_min = np.min(wavelength_range)
    wavelengths = np.linspace(wavelength_min, wavelength_max, Nfreqs)
    
    # ref is angle x wavelength
    # ref[i][j]: i-th angle, j-th wavelength
    
    plt.figure()
    plt.plot(angles, ref)
    
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Reflectivity")
    plt.title("Reflectivity vs Angle for Different Wavelengths")
    plt.grid(True)
    plt.legend()
    plt.show()
    '''

    return phase, gh_shift, ref, angles
                              
                             
def plot_compare(d1, d2, start_x, end_x, steps, steps2, title, label1, label2):
    plt.figure()
    plt.title(title)
    array1 = np.linspace(start_x, end_x, steps)
    array2 = np.linspace(start_x, end_x, steps2)
    plt.plot(array, d1, label=label1)
    plt.plot(array2, d2, label=label2)
    plt.show()
    
def figsave(angles, values, title, x_label, y_label, folder='investigations'):
    """
    Save data and plot into a specified folder.

    Parameters
    ----------
    angles : array-like
        The angle values for the x-axis.
    values : array-like
        The corresponding reflectivity (or other) values for the y-axis.
    title : str
        The title of the plot (will also be used as part of the filename).
    x_label : str
        The label for the x-axis.
    y_label : str
        The label for the y-axis.
    folder : str, optional
        The folder name where the files will be saved (default is 'investigations').

    Returns
    -------
    None
    """
    # Ensure the output directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Convert data to a DataFrame for easy CSV export
    df = pd.DataFrame({
        x_label: angles,
        y_label: values
    })
    
    # Create a filename-friendly version of the title by replacing spaces
    filename_base = title.replace(" ", "_")
    
    # Save the DataFrame to CSV
    csv_path = os.path.join(folder, f"{filename_base}.csv")
    df.to_csv(csv_path, index=False)
    
    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(angles, values)
    
    # Save the plot as a PDF
    pdf_path = os.path.join(folder, f"{filename_base}.pdf")
    plt.savefig(pdf_path)
    plt.show()
    plt.close()    
                                 
def get_intensity(data):
    Ex = data.Ex 
    Ey = data.Ey
    Ez = data.Ez
    E = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    Int = np.abs(E)**2

    return Int
    
def figsave2(d1, d2, start_x, end_x, steps, steps2, title, label1, label2, folder='investigations'):
    """
    Plot two datasets for comparison, save the data and the plot.
    
    Parameters
    ----------
    d1 : array-like
        The first set of values to plot.
    d2 : array-like
        The second set of values to plot.
    start_x : float
        The start value of the x-range.
    end_x : float
        The end value of the x-range.
    steps : int
        The number of steps (points) for the first dataset.
    steps2 : int
        The number of steps (points) for the second dataset.
    title : str
        The title of the plot (will also be used as part of the filename).
    label1 : str
        The label for the first dataset in the legend.
    label2 : str
        The label for the second dataset in the legend.
    folder : str, optional
        The folder name where the files will be saved (default is 'investigations').

    Returns
    -------
    None
    """
    # Ensure the output directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Create the x arrays for both datasets
    x1 = np.linspace(start_x, end_x, steps)
    x2 = np.linspace(start_x, end_x, steps2)

    # Create a DataFrame that holds both datasets
    # We'll have columns: X1, D1, X2, D2
    df = pd.DataFrame({
        "X1": x1,
        label1: d1,
        "X2": x2,
        label2: d2
    })

    # Create a filename-friendly version of the title
    filename_base = title.replace(" ", "_")
    
    # Save the data as a CSV
    csv_path = os.path.join(folder, f"{filename_base}.csv")
    df.to_csv(csv_path, index=False)
    
    # Plot the data
    plt.figure(figsize=(8,6))
    plt.title(title)
    plt.plot(x1, d1, label=label1)
    plt.plot(x2, d2, label=label2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # Save the plot as a PDF
    pdf_path = os.path.join(folder, f"{filename_base}.pdf")
    plt.savefig(pdf_path)

    # Now show the plot
    plt.show()
    
    # Close the figure
    plt.close()

#remake with new multi wavelength
def sens_scan(
    wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
     #hole material
    prism_material,
    start_sample,
    end_sample,
    sample_step,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    start_angle,
    end_angle,
    steps,
    sim_height):
    
    indices = np.linspace(start_sample, end_sample, sample_step)
    
    make_sim, hole_bot, hole_side, freq0 = set_nano_sim(wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    sim_height)
    
    central_wavelength=np.mean(wavelength_range)
    angles = np.linspace(start_angle, end_angle ,steps)

        
    #plot one graph to get pricing and plot ---------------------------------------------------------------------------------------
    nano_sim = make_sim(angle=angles[0])
    print("Plotting for angle", angles[0])
    # visualize geometry
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    nano_sim.plot(z=hole_bot, ax=ax1)
    nano_sim.plot(y=hole_side, ax=ax2)

    plt.show()
    
    task_id = web.upload(nano_sim, task_name="Nanohole Array")
    estimated_cost = web.estimate_cost(task_id=task_id)
    print("This simulation will cost", estimated_cost, "flex credits")
    print(steps, "simulations will cost", estimated_cost*steps, "flex credits")
    
    input("Proceed?")
    
    
    angles = np.linspace(start_angle, end_angle, steps)
    p_reflectivities = np.zeros(steps)
    phases = np.zeros(steps)
    phases2 = np.zeros(steps)

    sims = {f"l={l:.2f}": make_sim(l) for l in angles}
    batch = web.Batch(simulations=sims, verbose=True)

    batch_results = batch.run(path_dir="data")
    reflectivity_batch = []
    phase_batch_y = []
    phase_batch_z= []
    

    for task_name, sim_data in batch_results.items():
        trans_data = sim_data['T']
        ref_data = sim_data['R']
        phase_data = sim_data['phase']
        coupling = sim_data['side']

        
        
        phases_x, phases_y, phases_z = get_phase(phase_data)
        reflectivities = ref_multi(trans_data, ref_data, freq0)
        reflectivity_batch.append(reflectivities)
        phase_batch_y.append(phases_y)
        phase_batch_z.append(phases_z)
      
    phase  = phase_batch_z 
    pz = np.array(phase)
    
    #unwrap the phase for a smooth curve
    uwpz = np.degrees(np.unwrap(pz))

    #interpolate to make a continous curve
    interp_phase = CubicSpline(angles,(uwpz))

    #differentiate the curve 
    phase_div = interp_phase.derivative(1)(angles)


    #calculate the lateral goos-hanchen shift
    A = -2*np.pi / central_wavelength
    gh_shift = (1/A)*phase_div*1e-6
    
    deg_phase_z = np.degrees(phase_batch_z)
    uphase = np.unwrap(deg_phase_z)
    
    deg_phase_y = np.degrees(phase_batch_y)
    uphase_y = np.unwrap(deg_phase_y)
    
    plt.figure()
    plt.title("Reflectivity against angles")
    plt.plot(angles, reflectivity_batch)
    plt.show()
    
    plt.figure()
    plt.title("Point phase against angles z")
    plt.plot(angles, phase_batch_z)
    plt.show()
    
    plt.figure()
    plt.title("Lateral shift agains angles")
    plt.plot(angles, gh_shift)
    plt.show()



    return phase, gh_shift, reflectivity_batch, angles

#BELOW IS THE POROUS FUNCTION CREATION 

def create_ellipsoid(a, b, c, u_segments, v_segments):
    # initialize empty lists for vertices and faces
    vertices = []
    faces = []

    # create vertices
    for i in range(u_segments + 1):
        theta = i * np.pi / u_segments  # angle for the latitude (0 to pi)

        for j in range(v_segments + 1):
            phi = j * 2 * np.pi / v_segments  # angle for the longitude (0 to 2*pi)

            # compute vertex position using ellipsoidal equations
            x = a * np.sin(theta) * np.cos(phi)
            y = b * np.sin(theta) * np.sin(phi)
            z = c * np.cos(theta)

            vertices.append([x, y, z])

    # create faces
    for i in range(u_segments):
        for j in range(v_segments):
            # compute indices for vertices
            v1 = i * (v_segments + 1) + j
            v2 = (i + 1) * (v_segments + 1) + j
            v3 = (i + 1) * (v_segments + 1) + (j + 1)
            v4 = i * (v_segments + 1) + (j + 1)

            # create faces using the vertices
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])

    # create mesh using the generated vertices and faces
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def rand_coord(low_y, high_y,low_z,high_z, particles):
    #will generate random lateral coordinates between range
    #will generate random z coords make sure this is within film range
    
    N = particles  # number of random points
    x_coords = np.random.uniform(low_y, high_y, size=N)
    y_coords = np.random.uniform(low_y, high_y, size=N)
    z_coords = np.random.uniform(low_z, high_z, size=N)
    
    points = np.column_stack((x_coords, y_coords, z_coords))
    return(points)


def rand_ellipse(a_low, a_high, b_low, b_high, c_low, c_high, particles, u_segments, v_segments):
    ellipsoids = []
    for _ in range(particles):
        # Generate random values for a, b, c
        a = np.random.uniform(a_low, a_high)
        b = np.random.uniform(b_low, b_high)
        c = np.random.uniform(c_low, c_high)

        # Create the ellipsoid and add it to the list
        ellipsoid_mesh = create_ellipsoid(a, b, c, u_segments, v_segments)
        ellipsoids.append(ellipsoid_mesh)

    return ellipsoids
        
def pore_structure_rand(x_range, y_range, z_range,length_range, height_range, thick_range, pores, pore_material):
    x_low, x_high = x_range
    y_low, y_high = y_range
    z_low, z_high = z_range
    a_low, a_high = length_range
    b_low, b_high = height_range
    c_low, c_high = thick_range

    #defines resolution of ellipsoid creation 100 is good generally
    u_segments = v_segments = 100

    #generates random coords on grids for pores to take
    coords = rand_coord(y_low, y_high,z_low,z_high, pores)

    ellipses = rand_ellipse(a_low, a_high, b_low, b_high, c_low, c_high, pores, u_segments, v_segments)

    strucks=[]
    for i in range(pores):
        #define random rotation transform
        #need a transform about y (0,1,0)
        #then z z(0,0,1) for safety can also x
        x_val = np.random.uniform(0,2*np.pi)
        y_val = np.random.uniform(0,2*np.pi)
        z_val = np.random.uniform(0,2*np.pi)

        #make sure to rotate them about their own centres so they remain in film
        x_rot = trimesh.transformations.rotation_matrix(x_val, [coords[i][0],0,0])
        y_rot = trimesh.transformations.rotation_matrix(y_val, [0,coords[i][1],0])
        z_rot = trimesh.transformations.rotation_matrix(z_val, [0,0,coords[i][2]])
        #dissalowed rotation for now will fix param to height allowing confinement in metal
        
        ellipses[i].apply_translation(coords[i])
        ellipses[i].apply_transform(x_rot)
        ellipses[i].apply_transform(y_rot)
        ellipses[i].apply_transform(z_rot)
    
    
        
    
        ellipse = td.TriangleMesh.from_trimesh(ellipses[i])
        ellipse_str = td.Structure(geometry=ellipse, medium=pore_material)
        strucks.append(ellipse_str)
    
    listey=[]
    for i in range(len(strucks)):
        listey.append(strucks[i])

    return(listey)










#PORE SIM FUNC###############################################################################################################################
def set_pore_sim(
    wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    sim_height,
    length_range, 
    height_range,
    thick_range,
    pores):

    #SET PARAMATERS 
    
    #n_prism = np.sqrt(prism_material)
    stack_point =0
    sp = stack_point
    
    
    min_wavelength, max_wavelength = wavelength_range
    #frequency and wavelength paramaters
    central_wavelength =np.mean(wavelength_range)
    lda0 = central_wavelength  # central wavelength
    freq0 = td.C_0 / lda0  # central frequency
    ldas = np.linspace(min_wavelength, max_wavelength, Nfreqs)  # wavelength range
    freqs = td.C_0 / ldas  # frequency range
    fwidth = 0.5 * (np.max(freqs) - np.min(freqs))  # width of the source frequency range

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #define geometry for nanohole array
    h = height #depth of each hole 
    a = spacing #spacing between hole 
    d = diameter #diamater of hole
    t_base = t_base # thickness of base film

    material = metal_material
    hole_center = sp +h/2
    #CREATE UP TO FIVE HOLES FOR A UNIT CELL

    #here is the issue btw-----------------------------------------------------------------------------------------------------
    z_range =(sp,t_base)
              
    print(z_range)
    
    #define the pore structure
    pores = pore_structure_rand(x_range=(-a/2,a/2), y_range =(-a/2,a/2), z_range=z_range,
                    length_range=length_range, height_range=height_range, thick_range=thick_range, pores=pores
                    , pore_material = sample_material)

   
    hole_1 = td.Structure(
        geometry=td.Cylinder(center=(0, 0, hole_center), radius=d / 2, length=h), medium=background_material
    )
    
    hole_center
    hole_2 = td.Structure(
        geometry=td.Cylinder(center=(a / 2,  a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_3 = td.Structure(
        geometry=td.Cylinder(center=(a / 2, -1 * a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_4 = td.Structure(
        geometry=td.Cylinder(center=(-a / 2,  a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_5 = td.Structure(
        geometry=td.Cylinder(center=(-a / 2, -1 * a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )

    sphere = td.Structure(
        geometry=td.Sphere(center=(0, 0, sp-height), radius=d / 2), medium=background_material
    )
    
    # define the base plate structure
    base = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, sp),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, t_base)          # Upper bound: Top of the base at z = 0
    ),
    medium=material,
    name="MetalBase"
    )

    length_min = np.min(length_range)
    height_min = np.min(height_range)
    thick_min = np.min(thick_range)
    
    GoldMesh = td.MeshOverrideStructure(
        geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, sp),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, t_base)          # Upper bound: Top of the base at z = 0
    ),
        dl=(length_min/mesh_res, thick_min/mesh_res ,height_min/mesh_res),
        name ='GoldMesh'
)
  
####################

    prism = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, t_base),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, 100000)          # Upper bound: Top of the base at z = 0
    ),
    medium=prism_material,
    name="Prism"
    )

    sample = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, -10000000),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, sp)          # Upper bound: Top of the base at z = 0
    ),
    medium=sample_material,
    name="Sample"
    )

    '''    # Combine all holes into a single group
    hole_cell = [hole_1, hole_2, hole_3, hole_4, hole_5]
    holes = td.GeometryGroup(hole_cell)'''
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #allows user to choose Gaussian or plane wave source####################################################################################################################################################################

    #define source
    #plane_wave = td.PlaneWave(
    #source_time= td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    #size=(a, a, 0),
    #center=(0, 0, central_wavelength),
    #direction="+",
    #pol_angle=pol_angle,
    #angle_phi = np.pi/2,
    #angle_theta = np.radians(180-angle),
    #)

    

    #now define the flux monitors
    flux_monitor = td.FluxMonitor(
    center=[0, 0, central_wavelength*1.2], size=[td.inf, td.inf, 0], freqs=freqs, name="R"
    )

    flux_monitor2 = td.FluxMonitor(
    center=[0, 0, -1.2*central_wavelength +t_base], size=[td.inf, td.inf, 0], freqs=freqs, name="T"
    )


    monitor_field = td.FieldMonitor(
    center=[0,0,0],
    size= [0, np.inf,np.inf],
    freqs=freq0,
    name="field",
    )
    
    monitor_field2 = td.FieldMonitor(
    center=[d/2,0,0],
    size= [0, 0,np.inf],
    freqs=freq0,
    name="field2",
    )
    
    #point monitor for field phase
    r_phase_monitor = td.FieldMonitor(
    center=[0,0,central_wavelength*1.2],
    size= [0, 0,0],
    freqs=freq0,
    name="phase",
    )
    
    side = td.FieldMonitor(
    center=[0,0,sp+t_base/2],
    size= [0, np.inf,t_base],
    freqs=freq0,
    name="side",
    )
    top = td.FieldMonitor(
    center=[0,0,sp+h],
    size= [np.inf,np.inf,0],
    freqs=freq0,
    name="top",
    )
    bottom = td.FieldMonitor(
    center=[0,0,sp],
    size= [np.inf,np.inf,0],
    freqs=freq0,
    name="bottom",
    )
###########################################################################################################################################################################################################################
    
    
    def make_sim(angle):
        
        
        
        #define source
        plane_wave = td.PlaneWave(
        source_time= td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        size=(a, a, 0),
        center=(0, 0, central_wavelength),
        direction="+",
        pol_angle=pol_angle,
        angle_phi = np.pi/2,
        angle_theta = np.radians(180-angle),
        )
        
        run_time = run_t / freq0  # simulation run time

        # define simulation domain size
        sim_size = (a,  a,  sim_height*central_wavelength)
        # create the Bloch boundaries
        bloch_x = td.Boundary.bloch_from_source(
        source=plane_wave, domain_size=sim_size[0], axis=0, medium=prism_material
        )
        bloch_y = td.Boundary.bloch_from_source(
        source=plane_wave, domain_size=sim_size[1], axis=1, medium=prism_material
        )

        bspec_bloch = td.BoundarySpec(x=bloch_x, y=bloch_y, z=td.Boundary.pml())
        # Calculate wavevector components and Bloch wavevectors


        grid_spec = td.GridSpec.auto(
        min_steps_per_wvl = grid_res,
        override_structures = [GoldMesh],
        wavelength=lda0,
        )
        
        
            
        if hole == 'one':
            structures=[ base, hole_1, prism, sample]
        elif hole == 'zero':
            structures = [base, prism, sample]
        elif hole == 'five':
            structures = [base,hole_1, hole_2, hole_3, hole_4, hole_5, prism, sample]
        elif hole == 'spherical':
            structures=[base,sphere,prism,sample]
        elif hole=='pores':
            
            structures =[base,sample] + pores + [prism] #[base] + pores + [prism,sample]
            #structures.append(base)
            #structures.append(prism)
            #structures.append(sample)

        
        sim = td.Simulation(
        center=(0, 0, 0),
        size=sim_size,
        grid_spec=grid_spec,
        structures = structures,
        sources=[plane_wave],
        monitors=[flux_monitor, monitor_field, flux_monitor2, monitor_field2, r_phase_monitor, side, bottom, top],
        run_time=run_time,
        boundary_spec=bspec_bloch, 
        #symmetry=(1, -1, 0),
        #shutoff=1e-7,  # reducing the default shutoff level
        )
    
        return sim

        #ax = sim.plot(z=h / 2)
        #sim.plot_grid(z=h / 2, ax=ax)

    hole_bot = sp +h
    hole_side = 0
        
        
    
    return make_sim, hole_bot, hole_side, freq0,

def pore_scan(
    wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    start_angle,
    end_angle,
    steps,
    sim_height,
    length_range,
    height_range,
    thick_range,
    pores):
    
    
    make_sim, hole_bot, hole_side, freq0 = set_pore_sim(wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    sim_height,
    length_range,
    height_range,
    thick_range,
    pores)
    
    central_wavelength=np.mean(wavelength_range)
    angles = np.linspace(start_angle, end_angle ,steps)

        
    #plot one graph to get pricing and plot ---------------------------------------------------------------------------------------
    nano_sim = make_sim(angle=angles[0])
    print("Plotting for angle", angles[0])
    # visualize geometry
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Top-Left Subplot: Z-axis slices
    axs[0, 0].set_title('Z-axis Slices')
    nano_sim.plot(z=hole_bot, ax=axs[0, 0], label='Hole Bottom')
    nano_sim.plot(z=t_base, ax=axs[0, 0], label='T Base')
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('X-axis')
    axs[0, 0].set_ylabel('Y-axis')
    
    # Top-Right Subplot: Y-axis slice at hole_side
    axs[0, 1].set_title('Y-axis Slice')
    nano_sim.plot(y=hole_side,  ax=axs[0, 1], label='Hole Side')
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('X-axis')
    axs[0, 1].set_ylabel('Z-axis')
    axs[0,1].set_ylim(0,t_base)
    
    # Bottom-Left Subplot: X=0 Plane
    axs[1, 0].set_title('X=0 Plane')
    nano_sim.plot(x=0, ax=axs[1, 0], label='X=0 Slice')
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Y-axis')
    axs[1, 0].set_ylabel('Z-axis')
    axs[1,0].set_ylim(0,t_base)
    
    # Bottom-Right Subplot: Y=0 Plane
    axs[1, 1].set_title('Y=0 Plane')
    nano_sim.plot(y=0, ax=axs[1, 1], label='Y=0 Slice')
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('X-axis')
    axs[1, 1].set_ylabel('Z-axis')
    
    # Adjust layout for better spacing
    plt.tight_layout()
        #nano_sim.plot_3d()
    
    plt.show()
    
    task_id = web.upload(nano_sim, task_name="Nanohole Array")
    estimated_cost = web.estimate_cost(task_id=task_id)
    print("This simulation will cost", estimated_cost, "flex credits")
    print(steps, "simulations will cost", estimated_cost*steps, "flex credits")
    
    input("Proceed?")
    
    
    angles = np.linspace(start_angle, end_angle, steps)
    p_reflectivities = np.zeros(steps)
    phases = np.zeros(steps)
    phases2 = np.zeros(steps)

    sims = {f"l={l:.2f}": make_sim(l) for l in angles}
    batch = web.Batch(simulations=sims, verbose=True)

    batch_results = batch.run(path_dir="data")
    reflectivity_batch = []
    phase_batch_y = []
    phase_batch_z= []
    

    for task_name, sim_data in batch_results.items():
        trans_data = sim_data['T']
        ref_data = sim_data['R']
        phase_data = sim_data['phase']
        
        phases_y, phases_z = get_phase(phase_data)

        #calcualtes array of reflectivities at different wavelenths for each angle point
        reflectivities = reflectivity(trans_data, ref_data, freq0)

        #print(reflectivities)
            
        reflectivity_batch.append(reflectivities)
        phase_batch_y.append(phases_y)
        phase_batch_z.append(phases_z)

    #print(reflectivity_batch)
    ref = reflectivity_batch
    phasez  = phase_batch_z 
    phase = np.array(phasez)
    pz = phase
    
    #unwrap the phase for a smooth curve
    uwpz = np.degrees(np.unwrap(pz))

    #interpolate to make a continous curve
    interp_phase = CubicSpline(angles,(uwpz))

    #differentiate the curve 
    phase_div = interp_phase.derivative(1)(angles)


    #calculate the lateral goos-hanchen shift
    A = -2*np.pi / central_wavelength
    gh_shift = (1/A)*phase_div*1e-6
    
    deg_phase_z = np.degrees(phase_batch_z)
    uphase = np.unwrap(deg_phase_z)
    
    deg_phase_y = np.degrees(phase_batch_y)
    uphase_y = np.unwrap(deg_phase_y)
    
    #plt.figure()
    #plt.title("Reflectivity against angles")
    #plt.plot(angles, reflectivity_batch)
    #plt.show()
    '''
    plt.figure()
    plt.title("Point phase against angles z")
    plt.plot(angles, phase_batch_z)
    plt.show()
    
    plt.figure()
    plt.title("Lateral shift agains angles")
    plt.plot(angles, gh_shift)
    plt.show()

    ref = reflectivity_batch

    wavelength_max = np.max(wavelength_range)
    wavelength_min = np.min(wavelength_range)
    wavelengths = np.linspace(wavelength_min, wavelength_max, Nfreqs)
    
    # ref is angle x wavelength
    # ref[i][j]: i-th angle, j-th wavelength
    
    plt.figure()
    plt.plot(angles, ref)
    
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Reflectivity")
    plt.title("Reflectivity vs Angle for Different Wavelengths")
    plt.grid(True)
    plt.legend()
    plt.show()
    '''

    return phase, gh_shift, ref, angles



#DELEte BELOW HONESTLY I HATE IT

########################################################################################################################################################
#PORE SIM FUNC###############################################################################################################################
def set_pore_sim2(
    wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    sim_height,
    length_range, 
    height_range,
    thick_range,
    pores):

    #SET PARAMATERS 
    
    #n_prism = np.sqrt(prism_material)
    stack_point =0
    sp = stack_point
    
    
    min_wavelength, max_wavelength = wavelength_range
    #frequency and wavelength paramaters
    central_wavelength =np.mean(wavelength_range)
    lda0 = central_wavelength  # central wavelength
    freq0 = td.C_0 / lda0  # central frequency
    ldas = np.linspace(min_wavelength, max_wavelength, Nfreqs)  # wavelength range
    freqs = td.C_0 / ldas  # frequency range
    fwidth = 0.5 * (np.max(freqs) - np.min(freqs))  # width of the source frequency range

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #define geometry for nanohole array
    h = height #depth of each hole 
    a = spacing #spacing between hole 
    d = diameter #diamater of hole
    t_base = t_base # thickness of base film

    material = metal_material
    hole_center = sp +h/2
    #CREATE UP TO FIVE HOLES FOR A UNIT CELL

    #here is the issue btw-----------------------------------------------------------------------------------------------------
    z_range =(sp,t_base- (np.max(thick_range)/2))
              
    print(z_range)
    
    #define the pore structure
    pores = pore_structure_rand(x_range=(-a/2,a/2), y_range =(-a/2,a/2), z_range=z_range,
                    length_range=length_range, height_range=height_range, thick_range=thick_range, pores=pores
                    , pore_material = sample_material)

   
    hole_1 = td.Structure(
        geometry=td.Cylinder(center=(0, 0, hole_center), radius=d / 2, length=h), medium=background_material
    )
    
    hole_center
    hole_2 = td.Structure(
        geometry=td.Cylinder(center=(a / 2,  a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_3 = td.Structure(
        geometry=td.Cylinder(center=(a / 2, -1 * a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_4 = td.Structure(
        geometry=td.Cylinder(center=(-a / 2,  a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_5 = td.Structure(
        geometry=td.Cylinder(center=(-a / 2, -1 * a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )

    sphere = td.Structure(
        geometry=td.Sphere(center=(0, 0, sp-height), radius=d / 2), medium=background_material
    )
    
    # define the base plate structure
    base = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, sp),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, t_base)          # Upper bound: Top of the base at z = 0
    ),
    medium=material,
    name="MetalBase"
    )

    length_min = np.min(length_range)
    height_min = np.min(height_range)
    thick_min = np.min(thick_range)
    
    GoldMesh = td.MeshOverrideStructure(
        geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, sp),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, t_base)          # Upper bound: Top of the base at z = 0
    ),
        dl=(length_min/mesh_res, thick_min/mesh_res ,height_min/mesh_res),
        name ='GoldMesh'
)
  
####################

    prism = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, t_base),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, 100000)          # Upper bound: Top of the base at z = 0
    ),
    medium=prism_material,
    name="Prism"
    )

    sample = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, -10000000),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, sp)          # Upper bound: Top of the base at z = 0
    ),
    medium=sample_material,
    name="Sample"
    )

    '''    # Combine all holes into a single group
    hole_cell = [hole_1, hole_2, hole_3, hole_4, hole_5]
    holes = td.GeometryGroup(hole_cell)'''
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #allows user to choose Gaussian or plane wave source####################################################################################################################################################################

    #define source
    #plane_wave = td.PlaneWave(
    #source_time= td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    #size=(a, a, 0),
    #center=(0, 0, central_wavelength),
    #direction="+",
    #pol_angle=pol_angle,
    #angle_phi = np.pi/2,
    #angle_theta = np.radians(180-angle),
    #)

    

    #now define the flux monitors
    flux_monitor = td.FluxMonitor(
    center=[0, 0, central_wavelength*0.11], size=[td.inf, td.inf, 0], freqs=freqs, name="R"
    )

    flux_monitor2 = td.FluxMonitor(
    center=[0, 0, -0.11*central_wavelength +t_base], size=[td.inf, td.inf, 0], freqs=freqs, name="T"
    )


    monitor_field = td.FieldMonitor(
    center=[0,0,0],
    size= [0, np.inf,np.inf],
    freqs=freq0,
    name="field",
    )
    
    monitor_field2 = td.FieldMonitor(
    center=[d/2,0,0],
    size= [0, 0,np.inf],
    freqs=freq0,
    name="field2",
    )
    
    #point monitor for field phase
    r_phase_monitor = td.FieldMonitor(
    center=[0,0,central_wavelength*0.11],
    size= [0, 0,0],
    freqs=freq0,
    name="phase",
    )
    
    side = td.FieldMonitor(
    center=[0,0,sp+t_base/2],
    size= [0, np.inf,t_base],
    freqs=freq0,
    name="side",
    )
    top = td.FieldMonitor(
    center=[0,0,sp+h],
    size= [np.inf,np.inf,0],
    freqs=freq0,
    name="top",
    )
    bottom = td.FieldMonitor(
    center=[0,0,sp],
    size= [np.inf,np.inf,0],
    freqs=freq0,
    name="bottom",
    )
###########################################################################################################################################################################################################################
    
    
    def make_sim2(angle):
        
        
        
        #define source
        plane_wave = td.PlaneWave(
        source_time= td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        size=(a, a, 0),
        center=(0, 0, 0.1*central_wavelength),
        direction="+",
        pol_angle=pol_angle,
        angle_phi = np.pi/2,
        angle_theta = np.radians(180-angle),
        )
        
        run_time = run_t / freq0  # simulation run time

        # define simulation domain size
        sim_size = (a,  a,  sim_height*central_wavelength)
        # create the Bloch boundaries
        bloch_x = td.Boundary.bloch_from_source(
        source=plane_wave, domain_size=sim_size[0], axis=0, medium=prism_material
        )
        bloch_y = td.Boundary.bloch_from_source(
        source=plane_wave, domain_size=sim_size[1], axis=1, medium=prism_material
        )

        bspec_bloch = td.BoundarySpec(x=bloch_x, y=bloch_y, z=td.Boundary.pml())
        # Calculate wavevector components and Bloch wavevectors


        grid_spec = td.GridSpec.auto(
        min_steps_per_wvl = grid_res,
        override_structures = [GoldMesh],
        wavelength=lda0,
        )
        
        
            
        if hole == 'one':
            structures=[ base, hole_1, prism, sample]
        elif hole == 'zero':
            structures = [base, prism, sample]
        elif hole == 'five':
            structures = [base,hole_1, hole_2, hole_3, hole_4, hole_5, prism, sample]
        elif hole == 'spherical':
            structures=[base,sphere,prism,sample]
        elif hole=='pores':
            
            structures = pores +[base, prism,sample]
            #structures.append(base)
            #structures.append(prism)
            #structures.append(sample)

        
        sim = td.Simulation(
        center=(0, 0, 0),
        size=sim_size,
        grid_spec=grid_spec,
        structures = structures,
        sources=[plane_wave],
        monitors=[flux_monitor, monitor_field, flux_monitor2, r_phase_monitor, side, bottom, top],
        run_time=run_time,
        boundary_spec=bspec_bloch, 
        #symmetry=(1, -1, 0),
        #shutoff=1e-7,  # reducing the default shutoff level
        )
    
        return sim

        #ax = sim.plot(z=h / 2)
        #sim.plot_grid(z=h / 2, ax=ax)

    hole_bot = sp +h
    hole_side = 0
        
        
    
    return make_sim2, hole_bot, hole_side, freq0,

def pore_scan2(
    wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    start_angle,
    end_angle,
    steps,
    sim_height,
    length_range,
    height_range,
    thick_range,
    pores):
    
    
    make_sim, hole_bot, hole_side, freq0 = set_pore_sim2(wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    sim_height,
    length_range,
    height_range,
    thick_range,
    pores)
    
    central_wavelength=np.mean(wavelength_range)
    angles = np.linspace(start_angle, end_angle ,steps)

        
    #plot one graph to get pricing and plot ---------------------------------------------------------------------------------------
    nano_sim = make_sim(angle=angles[0])
    print("Plotting for angle", angles[0])
    # visualize geometry
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    nano_sim.plot(z=hole_bot, ax=ax1)
    nano_sim.plot(y=hole_side, ax=ax2)

    plt.show()
    
    task_id = web.upload(nano_sim, task_name="Nanohole Array")
    estimated_cost = web.estimate_cost(task_id=task_id)
    print("This simulation will cost", estimated_cost, "flex credits")
    print(steps, "simulations will cost", estimated_cost*steps, "flex credits")
    
    input("Proceed?")
    
    
    angles = np.linspace(start_angle, end_angle, steps)
    p_reflectivities = np.zeros(steps)
    phases = np.zeros(steps)
    phases2 = np.zeros(steps)

    sims = {f"l={l:.2f}": make_sim(l) for l in angles}
    batch = web.Batch(simulations=sims, verbose=True)

    batch_results = batch.run(path_dir="data")
    reflectivity_batch = []
    phase_batch_y = []
    phase_batch_z= []
    

    for task_name, sim_data in batch_results.items():
        trans_data = sim_data['T']
        ref_data = sim_data['R']
        phase_data = sim_data['phase']
        
        phases_y, phases_z = get_phase(phase_data)

        #calcualtes array of reflectivities at different wavelenths for each angle point
        reflectivities = reflectivity(trans_data, ref_data, freq0)

        #print(reflectivities)
            
        reflectivity_batch.append(reflectivities)
        phase_batch_y.append(phases_y)
        phase_batch_z.append(phases_z)

    #print(reflectivity_batch)
    ref = reflectivity_batch
    phasez  = phase_batch_z 
    phase = np.array(phasez)
    pz = phase
    
    #unwrap the phase for a smooth curve
    uwpz = np.degrees(np.unwrap(pz))

    #interpolate to make a continous curve
    interp_phase = CubicSpline(angles,(uwpz))

    #differentiate the curve 
    phase_div = interp_phase.derivative(1)(angles)


    #calculate the lateral goos-hanchen shift
    A = -2*np.pi / central_wavelength
    gh_shift = (1/A)*phase_div*1e-6
    
    deg_phase_z = np.degrees(phase_batch_z)
    uphase = np.unwrap(deg_phase_z)
    
    deg_phase_y = np.degrees(phase_batch_y)
    uphase_y = np.unwrap(deg_phase_y)
    
    #plt.figure()
    #plt.title("Reflectivity against angles")
    #plt.plot(angles, reflectivity_batch)
    #plt.show()
    '''
    plt.figure()
    plt.title("Point phase against angles z")
    plt.plot(angles, phase_batch_z)
    plt.show()
    
    plt.figure()
    plt.title("Lateral shift agains angles")
    plt.plot(angles, gh_shift)
    plt.show()

    ref = reflectivity_batch

    wavelength_max = np.max(wavelength_range)
    wavelength_min = np.min(wavelength_range)
    wavelengths = np.linspace(wavelength_min, wavelength_max, Nfreqs)
    
    # ref is angle x wavelength
    # ref[i][j]: i-th angle, j-th wavelength
    
    plt.figure()
    plt.plot(angles, ref)
    
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Reflectivity")
    plt.title("Reflectivity vs Angle for Different Wavelengths")
    plt.grid(True)
    plt.legend()
    plt.show()
    '''

    return phase, gh_shift, ref, angles
