import numpy as np
import os
import glob
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fftn, ifftn

def read_vtk_athena_v2(filename):
    with open(filename, 'rb') as f:
        content = f.read()
    def get_meta(key, default=None):
        pos = content.find(key.encode())
        if pos == -1: return default
        end = content.find(b'\n', pos)
        line = content[pos:end].decode()
        return line.split()[1:]
    dims = [int(x) for x in get_meta('DIMENSIONS')]
    origin = [float(x) for x in get_meta('ORIGIN')]
    spacing = [float(x) for x in get_meta('SPACING')]
    nx, ny, nz = dims[0]-1, dims[1]-1, dims[2]-1
    num_cells = nx * ny * nz
    data = {}
    for var in ['dens', 'velx', 'vely', 'velz']:
        marker = f'SCALARS {var} float'.encode()
        pos = content.find(marker)
        if pos == -1: continue
        data_start = content.find(b'default', pos) + 8
        var_data = np.frombuffer(content[data_start:data_start + num_cells*4], dtype='>f4').copy()
        data[var] = var_data.reshape((nz, ny, nx)) 
    return data, nx, ny, nz, spacing, origin

def spectral_filter(v, n_max=3):
    nz, ny, nx = v.shape
    v_hat = fftn(v)
    kz = np.fft.fftfreq(nz, 1/nz)
    ky = np.fft.fftfreq(ny, 1/ny)
    kx = np.fft.fftfreq(nx, 1/nx)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    K_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
    mask = K_mag <= n_max
    return ifftn(v_hat * mask).real

def compute_q_criterion(vx, vy, vz, dx):
    dvz, dvy, dvx = np.gradient(vx, dx, dx, dx)
    L13, L12, L11 = dvz, dvy, dvx
    dvz, dvy, dvx = np.gradient(vy, dx, dx, dx)
    L23, L22, L21 = dvz, dvy, dvx
    dvz, dvy, dvx = np.gradient(vz, dx, dx, dx)
    L33, L32, L31 = dvz, dvy, dvx
    trL2 = (L11*L11 + L12*L21 + L13*L31 +
            L21*L12 + L22*L22 + L23*L32 +
            L31*L13 + L32*L23 + L33*L33)
    return -0.5 * trL2

def main():
    data_dir = "/Users/dtgamage/Benchmark_Copilot/raw_data_tdad"
    vtk_files = sorted(glob.glob(os.path.join(data_dir, "*.vtk")))
    n_tracers = 500 
    sub_steps = 2
    dt_snapshot = 0.1
    dt = dt_snapshot / sub_steps
    data, nx, ny, nz, spacing, origin = read_vtk_athena_v2(vtk_files[0])
    dx = spacing[0]
    z_coords = np.linspace(origin[2] + dx/2, origin[2] + dx*(nz-0.5), nz)
    y_coords = np.linspace(origin[1] + dx/2, origin[1] + dx*(ny-0.5), ny)
    x_coords = np.linspace(origin[0] + dx/2, origin[0] + dx*(nx-0.5), nx)
    np.random.seed(42)
    tracer_pos = np.random.rand(n_tracers, 3) - 0.5 
    initial_pos = tracer_pos.copy()
    unwrapped_pos = tracer_pos.copy()
    v_old = [data['velx'], data['vely'], data['velz']]
    msd_list, msd_para_list, msd_perp_list = [], [], []
    q_residence = np.zeros(n_tracers)
    for i in range(len(vtk_files)-1):
        data_next, _, _, _, _, _ = read_vtk_athena_v2(vtk_files[i+1])
        v_next = [data_next['velx'], data_next['vely'], data_next['velz']]
        v_ls_now = [spectral_filter(v_old[j]) for j in range(3)]
        Q_field = compute_q_criterion(v_old[0], v_old[1], v_old[2], dx)
        q_func = RegularGridInterpolator((z_coords, y_coords, x_coords), Q_field, bounds_error=False, fill_value=0)
        for s in range(sub_steps):
            frac = (s + 0.5) / sub_steps
            v_curr = [(1-frac)*v_old[j] + frac*v_next[j] for j in range(3)]
            v_funcs = [RegularGridInterpolator((z_coords, y_coords, x_coords), v_curr[j], bounds_error=False, fill_value=None) for j in range(3)]
            pts = ((unwrapped_pos + 0.5) % 1.0 - 0.5)[:, [2, 1, 0]]
            v_at_p = np.column_stack([f(pts) for f in v_funcs])
            unwrapped_pos += v_at_p * dt
            q_residence += (q_func(pts) > 0).astype(float) * dt
        disp = unwrapped_pos - initial_pos
        v_ls_funcs = [RegularGridInterpolator((z_coords, y_coords, x_coords), v_ls_now[j], bounds_error=False, fill_value=None) for j in range(3)]
        vls_unit = np.column_stack([f(pts) for f in v_ls_funcs])
        vls_unit /= (np.linalg.norm(vls_unit, axis=1)[:, np.newaxis] + 1e-10)
        dp = np.sum(disp * vls_unit, axis=1)**2
        msd_list.append(np.mean(np.sum(disp**2, axis=1)))
        msd_para_list.append(np.mean(dp))
        msd_perp_list.append(np.mean(np.sum(disp**2, axis=1) - dp))
        v_old = v_next
        if i % 20 == 0: print(f"Snapshot {i} done")
    total_time = (len(vtk_files)-1) * dt_snapshot
    aniso = np.array(msd_para_list) / np.array(msd_perp_list)
    print(f"\nFinal MSD: {msd_list[-1]:.4f}\nAnisotropy (t>0.5): {np.mean(aniso[5:]):.3f}\nQ-Fraction: {np.mean(q_residence/total_time):.3f}")

if __name__ == "__main__":
    main()
