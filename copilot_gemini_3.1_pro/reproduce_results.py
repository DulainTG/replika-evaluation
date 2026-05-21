import os
import glob
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import ks_2samp, kurtosis, norm
from scipy.fft import fftn, ifftn, fftfreq

# Configuration
DATA_DIR = './raw_data_tdad'
OUTPUT_DIR = './results'
N_TRACERS = 8000
N_SUBSTEPS = 10
L = 1.0  # Domain side length
N_MODES_DRIVING = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_vtk_velocity_and_time(file_path):
    """ Reads VTK cell data (velx, vely, velz) and time from header. """
    # Read time from header manually
    time_val = 0.0
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            if "time=" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.startswith("time="):
                        time_val = float(p.split('=')[1]) if '=' in p and p.split('=')[1] != "" else float(parts[i+1])
                        break
                break

    mesh = pv.read(file_path)
    # The arrays are likely in point or cell data. 
    # Paper mentions 129^3 dimensions, so CELL_DATA is 128^3
    if 'velx' in mesh.cell_data:
        vx = mesh.cell_data['velx'].reshape((128, 128, 128), order='F')
        vy = mesh.cell_data['vely'].reshape((128, 128, 128), order='F')
        vz = mesh.cell_data['velz'].reshape((128, 128, 128), order='F')
    else:
        vx = mesh.point_data['velx'].reshape((129, 129, 129), order='F')[:-1,:-1,:-1]
        vy = mesh.point_data['vely'].reshape((129, 129, 129), order='F')[:-1,:-1,:-1]
        vz = mesh.point_data['velz'].reshape((129, 129, 129), order='F')[:-1,:-1,:-1]
    
    # We transpose to standard (x, y, z) depending on how pyvista arrays are flattened
    return time_val, np.stack([vx, vy, vz], axis=-1)

def compute_q_criterion(v_field, dx):
    """ Computes the Q-criterion for a 3D velocity field. """
    # v_field is (Nx, Ny, Nz, 3)
    # Gradients: dv_i/dx_j, shape is (3, 3, Nx, Ny, Nz)
    grad_v = np.zeros((3, 3, *v_field.shape[:3]))
    for i in range(3):
        grad_v[i, 0] = np.gradient(v_field[..., i], dx, axis=0, edge_order=2)
        grad_v[i, 1] = np.gradient(v_field[..., i], dx, axis=1, edge_order=2)
        grad_v[i, 2] = np.gradient(v_field[..., i], dx, axis=2, edge_order=2)
    
    Q = np.zeros(v_field.shape[:3])
    for x in range(v_field.shape[0]):
        for y in range(v_field.shape[1]):
            for z in range(v_field.shape[2]):
                grad = grad_v[:, :, x, y, z]
                Omega = 0.5 * (grad - grad.T)
                S = 0.5 * (grad + grad.T)
                Q[x, y, z] = 0.5 * (np.sum(Omega * Omega) - np.sum(S * S))
    return Q

def get_large_scale_velocity(v_field, N_modes=3):
    """ Spectral filter retaining only driving modes n <= 3. """
    v_ls = np.zeros_like(v_field)
    N = v_field.shape[0]
    freqs = fftfreq(N, d=1.0) * N
    kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    mask = k_mag <= N_modes
    
    for i in range(3):
        v_fft = fftn(v_field[..., i])
        v_fft_filtered = v_fft * mask
        v_ls[..., i] = np.real(ifftn(v_fft_filtered))
    return v_ls

def main():
    # 1. Gather files and sort them
    vtk_files = sorted(glob.glob(os.path.join(DATA_DIR, 'Turb.hydro_w.*.vtk')))
    if not vtk_files:
        print("No VTK files found.")
        return

    # Assuming uniform grid L=1
    N_cells = 128
    dx = L / N_cells
    x_coords = np.linspace(-L/2 + dx/2, L/2 - dx/2, N_cells)
    
    print("Reading first file to setup...")
    t0, v0 = read_vtk_velocity_and_time(vtk_files[0])
    vls0 = get_large_scale_velocity(v0, N_MODES_DRIVING)
    q0 = compute_q_criterion(v0, dx)

    # Setup particles
    np.random.seed(42)
    particles = np.random.uniform(-L/2, L/2, (N_TRACERS, 3))
    
    # Store trajectories
    n_snaps = len(vtk_files)
    trajectories = np.zeros((n_snaps, N_TRACERS, 3))
    trajectories[0] = particles
    times = np.zeros(n_snaps)
    times[0] = t0

    # Attributes along paths
    tracer_Q = np.zeros((n_snaps, N_TRACERS))
    tracer_VLS = np.zeros((n_snaps, N_TRACERS, 3))
    
    # Save Q at final snapshot for Figure 1
    Q_final_slice = None

    print(f"Integrating {N_TRACERS} tracers over {n_snaps} snapshots...")
    v_prev = v0
    t_prev = t0

    def get_interpolators(v, q, vls):
        interp_v = RegularGridInterpolator((x_coords, x_coords, x_coords), v, bounds_error=False, fill_value=None)
        interp_q = RegularGridInterpolator((x_coords, x_coords, x_coords), q, bounds_error=False, fill_value=None)
        interp_vls = RegularGridInterpolator((x_coords, x_coords, x_coords), vls, bounds_error=False, fill_value=None)
        return interp_v, interp_q, interp_vls
    
    interp_v0, interp_q0, interp_vls0 = get_interpolators(v0, q0, vls0)
    
    # Periodic bounds wrap
    def wrap(pos):
        return ((pos + L/2) % L) - L/2
    
    tracer_Q[0] = interp_q0(wrap(particles))
    tracer_VLS[0] = interp_vls0(wrap(particles))

    # Integration temporal loop
    for i in range(1, n_snaps):
        print(f"Processing snapshot {i}/{n_snaps}")
        t_next, v_next = read_vtk_velocity_and_time(vtk_files[i])
        vls_next = get_large_scale_velocity(v_next, N_MODES_DRIVING)
        q_next = compute_q_criterion(v_next, dx)
        times[i] = t_next
        
        interp_v1, interp_q1, interp_vls1 = get_interpolators(v_next, q_next, vls_next)
        
        dt_sub = (t_next - t_prev) / N_SUBSTEPS
        
        for sub in range(N_SUBSTEPS):
            frac = sub / N_SUBSTEPS
            
            def get_substep_velo(pos, frac_t):
                v_eval0 = interp_v0(wrap(pos))
                v_eval1 = interp_v1(wrap(pos))
                return (1 - frac_t) * v_eval0 + frac_t * v_eval1
            
            # RK4 Integration
            k1 = get_substep_velo(particles, frac)
            k2 = get_substep_velo(particles + 0.5 * dt_sub * k1, frac + 0.5/N_SUBSTEPS)
            k3 = get_substep_velo(particles + 0.5 * dt_sub * k2, frac + 0.5/N_SUBSTEPS)
            k4 = get_substep_velo(particles + dt_sub * k3, frac + 1.0/N_SUBSTEPS)
            
            particles = particles + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        trajectories[i] = particles
        tracer_Q[i] = interp_q1(wrap(particles))
        tracer_VLS[i] = interp_vls1(wrap(particles))
        
        v_prev, t_prev = v_next, t_next
        interp_v0, interp_q0, interp_vls0 = interp_v1, interp_q1, interp_vls1
        
        if i == n_snaps - 1:
            Q_final_slice = q_next[:, :, N_cells // 2]
            
    print("Integration complete. Processing statistics...")

    # Trajectories accumulated (unwrapped for displacement)
    # The MSD needs absolute unwrapped displacements over time lag.
    displacements = trajectories - trajectories[0:1] # (time, particles, 3)
    
    lag_times = times - times[0]
    
    # 3.1 MSD and scaling exponent
    msd = np.mean(np.sum(displacements**2, axis=-1), axis=1) # (time)
    
    alpha = np.zeros_like(msd)
    alpha[1:] = np.gradient(np.log(msd[1:]), np.log(lag_times[1:]))

    # Conditional MSD
    # Cohorts based on residence fraction (Q > 0)
    q_positive_fraction = np.mean(tracer_Q > 0, axis=0) # (particles)
    top_20_idx = np.argsort(q_positive_fraction)[-int(0.2*N_TRACERS):]
    bot_20_idx = np.argsort(q_positive_fraction)[:int(0.2*N_TRACERS)]

    msd_trapped = np.mean(np.sum(displacements[:, top_20_idx, :]**2, axis=-1), axis=1)
    msd_free = np.mean(np.sum(displacements[:, bot_20_idx, :]**2, axis=-1), axis=1)
    
    # 3.2 Anisotropy
    msd_para = np.zeros_like(msd)
    msd_perp = np.zeros_like(msd)
    for i in range(1, n_snaps):
        disp = displacements[i]
        v_ls_t = tracer_VLS[i] # (particles, 3)
        norm_vls = np.linalg.norm(v_ls_t, axis=-1, keepdims=True)
        v_hat = v_ls_t / (norm_vls + 1e-12)
        
        disp_para = np.sum(disp * v_hat, axis=-1)
        disp_para_vec = disp_para[:, None] * v_hat
        disp_perp_vec = disp - disp_para_vec
        
        msd_para[i] = np.mean(disp_para**2)
        msd_perp[i] = np.mean(np.sum(disp_perp_vec**2, axis=-1))

    anisotropy_ratio = msd_para / (msd_perp + 1e-12)
    anisotropy_ratio[0] = np.nan # t=0 undefined

    # 3.3 Autocorrelation of Q
    # Assuming stationary signal for each trajectory
    q_mean = np.mean(tracer_Q, axis=0)
    q_var = np.var(tracer_Q, axis=0)
    q_fluct = tracer_Q - q_mean
    
    # Simple inefficient but working autocorrelation
    q_autocorr = np.zeros(n_snaps)
    for lag in range(n_snaps):
        if lag == 0:
            q_autocorr[lag] = 1.0
        else:
            C = np.mean(q_fluct[:-lag] * q_fluct[lag:], axis=0)
            q_autocorr[lag] = np.mean(C / (q_var + 1e-12))
            
    # 3.4 PDF of displacements
    # At chosen target times (~0.5, 1.0, 2.0, 5.0, 9.0)
    # Map these actual physical times to indices
    target_times = [0.5, 1.0, 2.0, 5.0, 9.0]

    # Plots Generation
    print("Generating Figures...")

    # Figure 1: Q-criterion
    plt.figure(figsize=(6,5))
    q_plot = 10 * Q_final_slice / (np.max(np.abs(Q_final_slice)) + 1e-12) # normalize for colormap
    plt.imshow(q_plot, cmap='viridis', extent=[-L/2, L/2, -L/2, L/2])
    plt.title('Figure 1: Q-criterion at z=0 (Final Snapshot)')
    plt.colorbar(label='Normalized Q')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_1.png'))
    plt.close()

    # Figure 2: MSD
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].loglog(lag_times[1:], msd[1:], label='Ensemble MSD')
    ax[0].loglog(lag_times[1:], msd_trapped[1:], '--', label='Trapped')
    ax[0].loglog(lag_times[1:], msd_free[1:], ':.', label='Free')
    ax[0].axhline(y=L**2 / 6, color='k', linestyle=':', label='Saturation L^2/6')
    ax[0].set_xlabel('Time Lag t')
    ax[0].set_ylabel('MSD(t)')
    ax[0].legend()
    
    ax[1].plot(lag_times[1:], alpha[1:])
    ax[1].set_xscale('log')
    ax[1].set_ylim(0, 2.5)
    ax[1].axhline(y=1, color='k', linestyle=':')
    ax[1].set_xlabel('Time Lag t')
    ax[1].set_ylabel('Alpha(t)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_2.png'))
    plt.close()

    # Figure 3: Anisotropy
    plt.figure(figsize=(6,5))
    plt.semilogx(lag_times[1:], anisotropy_ratio[1:])
    plt.axhline(y=1.0, color='k', linestyle=':')
    plt.xlabel('Time Lag t')
    plt.ylabel('Anisotropy Ratio \lambda(t)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_3.png'))
    plt.close()

    # Figure 4: Q-autocorr
    plt.figure(figsize=(6,5))
    plt.plot(lag_times, q_autocorr)
    plt.axhline(y=np.exp(-1), color='r', linestyle='--', label='1/e threshold')
    plt.xlabel('Time Lag t')
    plt.ylabel('Autocorrelation of Q')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_4.png'))
    plt.close()

    # Figure 5: PDFs
    plt.figure(figsize=(8,6))
    for t_target in target_times:
        idx = np.argmin(np.abs(lag_times - t_target))
        if idx > 0 and idx < len(lag_times):
            disp = displacements[idx].flatten()  # 1D array of all components
            disp_norm = (disp - np.mean(disp)) / np.std(disp)
            # Evaluate PDF
            hist, bins = np.histogram(disp_norm, bins=50, density=True)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            plt.semilogy(bin_centers, hist, label=f't = {lag_times[idx]:.1f}')
    
    x_norm = np.linspace(-5, 5, 200)
    plt.semilogy(x_norm, norm.pdf(x_norm), 'k--', label='Gaussian')
    plt.ylim(1e-4, 1)
    plt.xlabel('Normalized Displacement')
    plt.ylabel('PDF')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_5.png'))
    plt.close()

    print("All required outputs written to:", OUTPUT_DIR)

if __name__ == '__main__':
    main()
