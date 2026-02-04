# gen_rf_cuda_v3.py
"""
P.mat 설정과 일관된 RF 데이터 생성.
- P.mat에서 모든 파라미터 로드 (fs, nSample, nChannel, nNumPulse 등)
- sources.mat에서 source 위치/타입 로드 (make_sources_from_slice_v2.py 결과)
- FUS TX delay를 고려한 cavitation emission timing
"""

import os
import numpy as np
import torch
import scipy.io as sio

# =============================================================================
# Configuration
# =============================================================================
P_MAT = r"./P.mat"
SOURCES_MAT = r"./sources.mat"
OUT_DIR = r"./RfData"
N_BURST = 30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Helpers
# =============================================================================
def mat_struct_to_dict(obj):
    """Recursively convert scipy.io.loadmat matlab struct objects to dict."""
    if isinstance(obj, np.ndarray) and obj.dtype == np.object_:
        if obj.size == 1:
            return mat_struct_to_dict(obj.item())
        return [mat_struct_to_dict(x) for x in obj.flat]
    if hasattr(obj, "_fieldnames"):
        d = {}
        for fn in obj._fieldnames:
            d[fn] = mat_struct_to_dict(getattr(obj, fn))
        return d
    return obj


def load_P(path):
    md = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return mat_struct_to_dict(md["P"])


def load_sources(path):
    md = sio.loadmat(path, squeeze_me=True)
    return {
        "src_x_mm": md["src_x_mm"].astype(np.float32).reshape(-1),
        "src_z_mm": md["src_z_mm"].astype(np.float32).reshape(-1),
        "src_amp": md["src_amp"].astype(np.float32).reshape(-1),
        "src_type": md["src_type"].astype(np.uint8).reshape(-1),
        "x0_mm": float(md["x0_mm"]),
        "z0_mm": float(md["z0_mm"]),
    }


# =============================================================================
# Pulse kernels
# =============================================================================
def make_stable_kernel(fs, f0, dur_us=15.0):
    """
    Stable cavitation: subharmonic + fundamental
    """
    dur = dur_us * 1e-6
    n = max(int(np.round(dur * fs)), 64)
    t = np.arange(n, dtype=np.float32) / fs
    win = np.hanning(n).astype(np.float32)
    
    # subharmonic (f0/2) + fundamental (f0)
    s = np.sin(2 * np.pi * (f0 / 2) * t) + 0.6 * np.sin(2 * np.pi * f0 * t)
    k = (s * win).astype(np.float32)
    k = k / (np.sqrt((k * k).sum()) + 1e-12)
    return k


def make_inertial_kernel(fs, dur_us=10.0, seed=None):
    """
    Inertial cavitation: broadband noise burst
    """
    rng = np.random.default_rng(seed)
    dur = dur_us * 1e-6
    n = max(int(np.round(dur * fs)), 64)
    win = np.hanning(n).astype(np.float32)
    
    raw = rng.standard_normal(n).astype(np.float32)
    k = (raw * win).astype(np.float32)
    k = k / (np.sqrt((k * k).sum()) + 1e-12)
    return k


def fft_conv1d_same(x, k):
    """
    FFT-based 1D convolution, same-length output.
    x: (C, T), k: (K,)
    """
    C, T = x.shape
    K = k.numel()
    L = T + K - 1
    
    X = torch.fft.rfft(x, n=L)
    Kf = torch.fft.rfft(k, n=L)
    Y = X * Kf
    y = torch.fft.irfft(Y, n=L)
    
    start = (K - 1) // 2
    return y[:, start:start + T]


# =============================================================================
# Main
# =============================================================================
def main():
    print(f"DEVICE: {DEVICE}")
    
    # -------------------------------------------------------------------------
    # 1. P.mat에서 모든 파라미터 로드
    # -------------------------------------------------------------------------
    print(f"\nLoading P.mat...")
    P = load_P(P_MAT)
    
    stRfInfo = P["CAV"]["stRfInfo"]
    FUS = P["FUS"]
    stTrans = P["stTrans"]
    
    # RF 파라미터
    c = float(stRfInfo["nSoundSpeed"])          # 1540 m/s
    nSample = int(stRfInfo["nSample"])          # 1280
    nCh = int(stRfInfo["nChannel"])             # 128
    fs = float(stRfInfo["nFs"])                 # 62.5 MHz
    f0 = float(stRfInfo["nFc"])                 # 15.625 MHz (center frequency)
    nPulse = int(FUS["nNumPulse"])              # 100
    
    print(f"  c = {c} m/s")
    print(f"  fs = {fs/1e6:.2f} MHz")
    print(f"  f0 = {f0/1e6:.3f} MHz")
    print(f"  nSample = {nSample}")
    print(f"  nChannel = {nCh}")
    print(f"  nNumPulse = {nPulse}")
    
    # Transducer element positions (meters)
    ele_x = np.array(stTrans["aElePos"], dtype=np.float32).reshape(-1)
    if ele_x.size != nCh:
        raise RuntimeError(f"aElePos len {ele_x.size} != nCh {nCh}")
    
    # Lens correction
    lens_m = float(stTrans.get("nLensCorr_m", 0.0))
    ele_z = np.full_like(ele_x, -lens_m, dtype=np.float32)
    
    print(f"  Element X range: {ele_x.min()*1000:.2f} ~ {ele_x.max()*1000:.2f} mm")
    print(f"  Lens correction: {lens_m*1000:.3f} mm")
    
    # -------------------------------------------------------------------------
    # 2. Sources 로드
    # -------------------------------------------------------------------------
    print(f"\nLoading sources.mat...")
    src = load_sources(SOURCES_MAT)
    
    src_x = (src["src_x_mm"] * 1e-3).astype(np.float32)  # mm -> m
    src_z = (src["src_z_mm"] * 1e-3).astype(np.float32)
    src_amp = src["src_amp"]
    src_type = src["src_type"]
    
    nSrc = src_x.size
    print(f"  N sources: {nSrc}")
    print(f"  X range: {src_x.min()*1000:.2f} ~ {src_x.max()*1000:.2f} mm")
    print(f"  Z range: {src_z.min()*1000:.2f} ~ {src_z.max()*1000:.2f} mm")
    print(f"  Focus center: x={src['x0_mm']:.2f} mm, z={src['z0_mm']:.2f} mm")
    
    # -------------------------------------------------------------------------
    # 3. Delay 계산
    # -------------------------------------------------------------------------
    # Cavitation은 self-emission이므로:
    #   signal_delay = source-to-element distance / c
    # 
    # FUS TX delay는 PCI beamformer가 처리하므로 여기서는 one-way만 계산
    
    # Move to torch
    ele_x_t = torch.from_numpy(ele_x).to(DEVICE)
    ele_z_t = torch.from_numpy(ele_z).to(DEVICE)
    src_x_t = torch.from_numpy(src_x).to(DEVICE)
    src_z_t = torch.from_numpy(src_z).to(DEVICE)
    src_amp_t = torch.from_numpy(src_amp).to(DEVICE)
    src_type_t = torch.from_numpy(src_type.astype(np.int64)).to(DEVICE)
    
    # Distance: (nSrc, nCh)
    dx = src_x_t[:, None] - ele_x_t[None, :]
    dz = src_z_t[:, None] - ele_z_t[None, :]
    dist = torch.sqrt(dx * dx + dz * dz) + 1e-9
    
    # Time-of-arrival (one-way)
    tau = dist / c  # seconds
    n0 = torch.round(tau * fs).to(torch.int64)  # sample index
    
    # Valid arrivals (within pulse buffer)
    valid = (n0 >= 0) & (n0 < nSample)
    
    # Attenuation (1/r)
    att = 1.0 / dist
    
    print(f"\n  Arrival sample range: {n0.min().item()} ~ {n0.max().item()}")
    print(f"  Valid arrivals: {valid.sum().item()} / {valid.numel()}")
    
    # -------------------------------------------------------------------------
    # 4. Prepare kernels
    # -------------------------------------------------------------------------
    k_stable = torch.from_numpy(make_stable_kernel(fs, f0, dur_us=15.0)).to(DEVICE)
    
    # Flatten indices for scatter
    ch_idx = torch.arange(nCh, device=DEVICE).view(1, -1).expand(nSrc, nCh)
    
    flat_ch = ch_idx[valid]
    flat_n0 = n0[valid]
    flat_att = att[valid]
    flat_amp = src_amp_t[:, None].expand(nSrc, nCh)[valid]
    flat_type = src_type_t[:, None].expand(nSrc, nCh)[valid]
    
    # Split stable / inertial
    mask_stable = (flat_type == 0)
    mask_inert = (flat_type == 1)
    
    idx_flat_st = (flat_ch[mask_stable] * nSample + flat_n0[mask_stable]).to(torch.int64)
    idx_flat_in = (flat_ch[mask_inert] * nSample + flat_n0[mask_inert]).to(torch.int64)
    
    base_val_st = (flat_amp[mask_stable] * flat_att[mask_stable]).to(torch.float32)
    base_val_in = (flat_amp[mask_inert] * flat_att[mask_inert]).to(torch.float32)
    
    print(f"\n  Stable events: {mask_stable.sum().item()}")
    print(f"  Inertial events: {mask_inert.sum().item()}")
    
    # -------------------------------------------------------------------------
    # 5. Generate bursts
    # -------------------------------------------------------------------------
    print(f"\nGenerating {N_BURST} bursts...")
    
    for b in range(1, N_BURST + 1):
        # Per-burst jitter
        jitter_st = 0.8 + 0.4 * torch.rand_like(base_val_st)
        jitter_in = 0.8 + 0.4 * torch.rand_like(base_val_in)
        
        # Impulse buffers
        imp_st = torch.zeros(nCh * nSample, device=DEVICE, dtype=torch.float32)
        imp_in = torch.zeros(nCh * nSample, device=DEVICE, dtype=torch.float32)
        
        imp_st.scatter_add_(0, idx_flat_st, base_val_st * jitter_st)
        imp_in.scatter_add_(0, idx_flat_in, base_val_in * jitter_in)
        
        imp_st = imp_st.view(nCh, nSample)
        imp_in = imp_in.view(nCh, nSample)
        
        # Convolve
        rf_st = fft_conv1d_same(imp_st, k_stable)
        
        k_in = torch.from_numpy(make_inertial_kernel(fs, dur_us=10.0, seed=b)).to(DEVICE)
        rf_in = fft_conv1d_same(imp_in, k_in)
        
        rf_pulse = rf_st + rf_in  # (nCh, nSample)
        
        # Tile across pulses with per-pulse gain variation
        gains = (0.95 + 0.10 * torch.rand(nPulse, device=DEVICE)).view(1, 1, nPulse)
        rf_tile = rf_pulse.unsqueeze(-1) * gains  # (nCh, nSample, nPulse)
        rf_all = rf_tile.reshape(nCh, nSample * nPulse)
        
        # Add sensor noise
        noise_level = 0.002 * rf_all.abs().mean()
        rf_all = rf_all + noise_level * torch.randn_like(rf_all)
        
        # Scale to int16
        mx = torch.max(torch.abs(rf_all)) + 1e-9
        scale = (0.8 * 32767.0) / mx
        rf_i16 = torch.clamp(rf_all * scale, -32768, 32767).to(torch.int16)
        
        # Save in column-major order for MATLAB: (L, nCh) Fortran order
        L = nSample * nPulse
        A = rf_i16.transpose(0, 1).contiguous()  # (L, nCh)
        A_cpu = A.cpu().numpy()
        data = np.ascontiguousarray(A_cpu.flatten(order="F"))
        
        fname = os.path.join(OUT_DIR, f"RfData_spc_{b:03d}.bin")
        data.tofile(fname)
        
        print(f"  [{b:03d}/{N_BURST}] {fname} (scale={float(scale):.3g})")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
