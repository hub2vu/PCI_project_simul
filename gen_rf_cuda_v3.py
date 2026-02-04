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
from scipy.interpolate import RegularGridInterpolator

# =============================================================================
# Configuration
# =============================================================================
# Use __file__ based paths for robustness
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
P_MAT = os.path.join(_SCRIPT_DIR, "P.mat")
SOURCES_MAT = os.path.join(_SCRIPT_DIR, "sources.mat")
OUT_DIR = os.path.join(_SCRIPT_DIR, "RfData")
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
    # 3. Delay 계산 (TX delay + RX delay를 빔포머와 일치시킴)
    # -------------------------------------------------------------------------
    # 빔포머에서: nDelay_m = (nTxDist + nRxDist) + nLensCorr - nOffset_m
    # RF 데이터도 동일한 delay 규칙을 따라야 빔포머가 올바른 위치에서 신호를 찾음
    #
    # Total delay = TX delay (FUS가 source에 도달하는 시간)
    #             + RX delay (source에서 element까지)
    #             + lens correction
    #             - acquisition offset

    # P.mat에서 추가 파라미터 로드
    CAV = P["CAV"]
    stG = CAV["stG"]

    startDepth = float(CAV.get("startDepth", 0))
    wavelength = float(stTrans.get("nWaveLength", c / f0))
    offset_m = startDepth * wavelength  # RF acquisition 시작 offset

    # TX delay matrix 로드 (있으면)
    mTxDelay_zx_m = None
    if "mTxDelay_zx_m" in CAV and CAV["mTxDelay_zx_m"] is not None:
        mTxDelay_zx_m = np.array(CAV["mTxDelay_zx_m"], dtype=np.float32)
        print(f"  mTxDelay_zx_m loaded: shape {mTxDelay_zx_m.shape}")
        print(f"  TX delay range: {mTxDelay_zx_m.min()*1000:.3f} ~ {mTxDelay_zx_m.max()*1000:.3f} mm")

    # Grid 좌표 (TX delay 보간용)
    aX = np.array(stG["aX"], dtype=np.float32).reshape(-1)
    aZ = np.array(stG["aZ"], dtype=np.float32).reshape(-1)

    print(f"\n  startDepth: {startDepth}")
    print(f"  wavelength: {wavelength*1000:.4f} mm")
    print(f"  offset_m: {offset_m*1000:.3f} mm")
    print(f"  lens_m: {lens_m*1000:.3f} mm")

    # Move to torch
    ele_x_t = torch.from_numpy(ele_x).to(DEVICE)
    ele_z_t = torch.from_numpy(ele_z).to(DEVICE)
    src_x_t = torch.from_numpy(src_x).to(DEVICE)
    src_z_t = torch.from_numpy(src_z).to(DEVICE)
    src_amp_t = torch.from_numpy(src_amp).to(DEVICE)
    src_type_t = torch.from_numpy(src_type.astype(np.int64)).to(DEVICE)

    # RX Distance: (nSrc, nCh) - source에서 각 element까지 거리
    dx = src_x_t[:, None] - ele_x_t[None, :]
    dz = src_z_t[:, None] - ele_z_t[None, :]
    rx_dist = torch.sqrt(dx * dx + dz * dz) + 1e-9

    # TX delay 계산: source 위치에서 보간 또는 근사
    if mTxDelay_zx_m is not None:
        # 각 source 위치에서 TX delay를 bilinear interpolation으로 보간
        # Grid 순서 확인: mTxDelay_zx_m[z_idx, x_idx]
        interp_tx = RegularGridInterpolator(
            (aZ, aX), mTxDelay_zx_m,
            method='linear', bounds_error=False, fill_value=None
        )

        # Source 위치에서 TX delay 보간
        src_points = np.stack([src_z, src_x], axis=1)  # (nSrc, 2): [z, x]
        tx_delay_per_src = interp_tx(src_points).astype(np.float32)  # (nSrc,) in meters

        # NaN 처리 (grid 밖의 source)
        tx_delay_per_src = np.nan_to_num(tx_delay_per_src, nan=float(aZ.mean()))

        tx_delay_t = torch.from_numpy(tx_delay_per_src).to(DEVICE)
        print(f"  TX delay per source: {tx_delay_t.min().item()*1000:.3f} ~ {tx_delay_t.max().item()*1000:.3f} mm")
    else:
        # TX delay가 없으면 source의 z 좌표를 TX delay로 근사 (plane wave 가정)
        print("  Warning: mTxDelay_zx_m not found, using source depth as TX delay approximation")
        tx_delay_t = src_z_t  # meters

    # Total delay 계산 (빔포머와 일치)
    # nDelay_m = (nTxDist + nRxDist) + nLensCorr - nOffset_m
    # TX delay는 각 source마다 동일 (element에 무관), RX delay는 (nSrc, nCh)
    total_delay_m = tx_delay_t[:, None] + rx_dist + lens_m - offset_m  # (nSrc, nCh)

    # Time-of-arrival -> sample index
    tau = total_delay_m / c  # seconds
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
    # Cavitation은 stochastic process이므로 각 pulse마다 독립적인 신호를 생성해야 함
    # 이것이 SVD 필터링이 제대로 작동하기 위한 핵심 요소임

    # Source activation probability per pulse (cavitation은 확률적)
    P_ACTIVE_STABLE = 0.7   # stable cavitation 활성화 확률
    P_ACTIVE_INERT = 0.5    # inertial cavitation 활성화 확률

    print(f"\nGenerating {N_BURST} bursts with pulse-to-pulse variation...")
    print(f"  P_active (stable): {P_ACTIVE_STABLE}, P_active (inertial): {P_ACTIVE_INERT}")

    for b in range(1, N_BURST + 1):
        # 각 pulse마다 독립적인 신호를 생성
        rf_all_pulses = []

        for p in range(nPulse):
            # Per-pulse random seed for reproducibility within burst
            pulse_seed = b * 1000 + p
            torch.manual_seed(pulse_seed)

            # Source activation: 각 pulse마다 다른 subset의 source가 활성화됨
            # 이것이 SVD가 cavitation을 분리할 수 있게 해주는 핵심
            active_st = torch.rand(mask_stable.sum().item(), device=DEVICE) < P_ACTIVE_STABLE
            active_in = torch.rand(mask_inert.sum().item(), device=DEVICE) < P_ACTIVE_INERT

            # Per-pulse jitter (timing + amplitude)
            jitter_st = 0.6 + 0.8 * torch.rand(mask_stable.sum().item(), device=DEVICE)
            jitter_in = 0.6 + 0.8 * torch.rand(mask_inert.sum().item(), device=DEVICE)

            # Apply activation mask
            val_st = base_val_st * jitter_st * active_st.float()
            val_in = base_val_in * jitter_in * active_in.float()

            # Impulse buffers
            imp_st = torch.zeros(nCh * nSample, device=DEVICE, dtype=torch.float32)
            imp_in = torch.zeros(nCh * nSample, device=DEVICE, dtype=torch.float32)

            imp_st.scatter_add_(0, idx_flat_st, val_st)
            imp_in.scatter_add_(0, idx_flat_in, val_in)

            imp_st = imp_st.view(nCh, nSample)
            imp_in = imp_in.view(nCh, nSample)

            # Convolve with kernels
            rf_st = fft_conv1d_same(imp_st, k_stable)

            # Inertial: 각 pulse마다 다른 random noise kernel
            k_in = torch.from_numpy(make_inertial_kernel(fs, dur_us=10.0, seed=pulse_seed)).to(DEVICE)
            rf_in = fft_conv1d_same(imp_in, k_in)

            rf_pulse = rf_st + rf_in  # (nCh, nSample)
            rf_all_pulses.append(rf_pulse)

        # Stack all pulses: (nCh, nSample, nPulse) -> (nCh, nSample*nPulse)
        rf_stack = torch.stack(rf_all_pulses, dim=2)  # (nCh, nSample, nPulse)
        rf_all = rf_stack.reshape(nCh, nSample * nPulse)

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
