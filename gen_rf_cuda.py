import os
import numpy as np
import torch
import scipy.io as sio

# ---------------------------
# Config
# ---------------------------
P_MAT = r"./P.mat"             # 너가 가진 P.mat 경로
SOURCES_MAT = r"./sources.mat" # python make_sources_from_slice.py 결과
OUT_DIR = r"./RfData"          # 여기에 RfData_spc_###.bin 생성
N_BURST = 30                   # 30장 생성
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# Helpers: MATLAB struct load (scipy loadmat)
# ---------------------------
def _mat_struct_to_dict(obj):
    """Recursively convert scipy.io.loadmat matlab struct objects to dict."""
    if isinstance(obj, np.ndarray) and obj.dtype == np.object_:
        if obj.size == 1:
            return _mat_struct_to_dict(obj.item())
        return [_mat_struct_to_dict(x) for x in obj.flat]
    # matlab struct from scipy has attribute _fieldnames
    if hasattr(obj, "_fieldnames"):
        d = {}
        for fn in obj._fieldnames:
            d[fn] = _mat_struct_to_dict(getattr(obj, fn))
        return d
    return obj

def load_P(P_path):
    md = sio.loadmat(P_path, squeeze_me=True, struct_as_record=False)
    Praw = md["P"]
    P = _mat_struct_to_dict(Praw)
    return P

def load_sources(src_path):
    md = sio.loadmat(src_path, squeeze_me=True, struct_as_record=False)
    # make_sources_from_slice.py 기준 키
    src_x_mm = md["src_x_mm"].astype(np.float32).reshape(-1)
    src_z_mm = md["src_z_mm"].astype(np.float32).reshape(-1)
    src_amp  = md["src_amp"].astype(np.float32).reshape(-1)
    src_type = md["src_type"].astype(np.uint8).reshape(-1)  # 1=inertial,0=stable
    return src_x_mm, src_z_mm, src_amp, src_type

# ---------------------------
# Build kernels (pulse-shape)
# ---------------------------
def make_stable_kernel(fs, f0, dur_us=15.0):
    """
    협대역 성분(예: subharmonic+fundamental)을 간단히 합성한 커널.
    길이 = dur_us 동안의 windowed sinusoid.
    """
    dur = dur_us * 1e-6
    n = int(np.round(dur * fs))
    n = max(n, 64)
    t = np.arange(n, dtype=np.float32) / fs
    win = np.hanning(n).astype(np.float32)

    # stable: subharmonic + fundamental (너 MATLAB 코드와 유사)
    s = np.sin(2*np.pi*(f0/2)*t) + 0.6*np.sin(2*np.pi*f0*t)
    k = (s * win).astype(np.float32)

    # 에너지 정규화(스케일 안정화)
    k = k / (np.sqrt((k*k).sum()) + 1e-12)
    return k

def make_inertial_kernel(fs, dur_us=10.0, seed=None):
    """
    광대역 짧은 noise burst 커널(간단 버전).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal
    else:
        noise = np.random.standard_normal

    dur = dur_us * 1e-6
    n = int(np.round(dur * fs))
    n = max(n, 64)
    win = np.hanning(n).astype(np.float32)
    k = (noise(n).astype(np.float32) * win).astype(np.float32)
    k = k / (np.sqrt((k*k).sum()) + 1e-12)
    return k

def fft_conv1d_same(x, k):
    """
    x: (C, T) torch float32
    k: (K,) torch float32
    return: (C, T) same-length linear convolution
    """
    C, T = x.shape
    K = k.numel()
    L = T + K - 1

    # rfft along time
    X = torch.fft.rfft(x, n=L)
    Kf = torch.fft.rfft(k, n=L)
    Y = X * Kf  # broadcasting over C
    y = torch.fft.irfft(Y, n=L)

    # crop center to "same"
    start = (K - 1)//2
    y_same = y[:, start:start+T]
    return y_same

# ---------------------------
# Main
# ---------------------------
def main():
    print("DEVICE:", DEVICE)

    P = load_P(P_MAT)
    src_x_mm, src_z_mm, src_amp, src_type = load_sources(SOURCES_MAT)

    # P 구조에서 파라미터 추출 (네가 찍어준 값과 일치)
    stRfInfo = P["CAV"]["stRfInfo"]
    FUS = P["FUS"]
    stTrans = P["stTrans"]

    c = float(stRfInfo["nSoundSpeed"])
    nSample = int(stRfInfo["nSample"])
    nCh = int(stRfInfo["nChannel"])
    fs = float(stRfInfo["nFs"])
    f0 = float(stRfInfo["nFc"])          # 여기선 예시로 수신 중심 주파수로 사용
    nPulse = int(FUS["nNumPulse"])

    # element positions (meters): aElePos is x only
    ele_x = np.array(stTrans["aElePos"], dtype=np.float32).reshape(-1)
    if ele_x.size != nCh:
        raise RuntimeError(f"aElePos len {ele_x.size} != nCh {nCh}")
    ele_z = np.zeros_like(ele_x, dtype=np.float32)

    # lens correction (optional)
    lens = float(stTrans.get("nLensCorr_m", 0.0))
    ele_z = ele_z - np.float32(lens)

    # sources (meters)
    src_x = (src_x_mm * 1e-3).astype(np.float32)
    src_z = (src_z_mm * 1e-3).astype(np.float32)

    # move to torch
    ele_x_t = torch.from_numpy(ele_x).to(DEVICE)
    ele_z_t = torch.from_numpy(ele_z).to(DEVICE)
    src_x_t = torch.from_numpy(src_x).to(DEVICE)
    src_z_t = torch.from_numpy(src_z).to(DEVICE)
    src_amp_t = torch.from_numpy(src_amp.astype(np.float32)).to(DEVICE)
    src_type_t = torch.from_numpy(src_type.astype(np.int64)).to(DEVICE)

    nSrc = src_x_t.numel()
    print(f"nSrc={nSrc}, nCh={nCh}, nSample={nSample}, nPulse={nPulse}, fs={fs/1e6:.2f}MHz, f0={f0/1e6:.3f}MHz")

    # Precompute distance & tau for one pulse window
    # dist: (nSrc, nCh)
    dx = src_x_t[:, None] - ele_x_t[None, :]
    dz = src_z_t[:, None] - ele_z_t[None, :]
    dist = torch.sqrt(dx*dx + dz*dz) + 1e-9
    tau = dist / c  # seconds

    # sample index within ONE pulse
    n0 = torch.round(tau * fs).to(torch.int64)  # (nSrc, nCh)

    # Keep only arrivals within the pulse buffer
    valid = (n0 >= 0) & (n0 < nSample)

    # amplitude model: (src_amp * 1/r) with small jitter will be per-burst
    att = 1.0 / dist  # (nSrc, nCh)

    # Kernels
    k_stable = torch.from_numpy(make_stable_kernel(fs, f0, dur_us=15.0)).to(DEVICE)
    # inertial kernel will be regenerated per burst for realism

    # Flatten indices helper
    ch_idx = torch.arange(nCh, device=DEVICE).view(1, -1).expand(nSrc, nCh)  # (nSrc, nCh)
    flat_ch = ch_idx[valid]
    flat_n0 = n0[valid]
    flat_att = att[valid]

    # also flatten source-dependent arrays aligned to valid mask
    # Need src_amp and src_type broadcast to (nSrc,nCh)
    flat_amp = src_amp_t[:, None].expand(nSrc, nCh)[valid]
    flat_type = src_type_t[:, None].expand(nSrc, nCh)[valid]

    # Split stable / inertial events
    mask_stable = (flat_type == 0)
    mask_inert  = (flat_type == 1)

    # Precompute base indices for scatter into (nCh*nSample)
    idx_flat_st = (flat_ch[mask_stable] * nSample + flat_n0[mask_stable]).to(torch.int64)
    idx_flat_in = (flat_ch[mask_inert]  * nSample + flat_n0[mask_inert]).to(torch.int64)

    base_val_st = (flat_amp[mask_stable] * flat_att[mask_stable]).to(torch.float32)
    base_val_in = (flat_amp[mask_inert]  * flat_att[mask_inert]).to(torch.float32)

    # Generate bursts
    for b in range(1, N_BURST+1):
        # jitter per burst (and slightly different between stable/inertial)
        jitter_st = (0.8 + 0.4*torch.rand_like(base_val_st))
        jitter_in = (0.8 + 0.4*torch.rand_like(base_val_in))

        # Impulse buffer for one pulse: (nCh*nSample,)
        imp_st = torch.zeros(nCh*nSample, device=DEVICE, dtype=torch.float32)
        imp_in = torch.zeros(nCh*nSample, device=DEVICE, dtype=torch.float32)

        # scatter_add into impulse map
        imp_st.scatter_add_(0, idx_flat_st, base_val_st * jitter_st)
        imp_in.scatter_add_(0, idx_flat_in, base_val_in * jitter_in)

        # reshape to (nCh, nSample)
        imp_st = imp_st.view(nCh, nSample)
        imp_in = imp_in.view(nCh, nSample)

        # convolve (FFT) within one pulse
        rf_st = fft_conv1d_same(imp_st, k_stable)

        k_in = torch.from_numpy(make_inertial_kernel(fs, dur_us=10.0, seed=b)).to(DEVICE)
        rf_in = fft_conv1d_same(imp_in, k_in)

        rf_pulse = rf_st + rf_in  # (nCh, nSample)

        # Tile across pulses (nPulse): add tiny per-pulse gain + noise for SVD realism
        gains = (0.95 + 0.10*torch.rand(nPulse, device=DEVICE, dtype=torch.float32)).view(1, 1, nPulse)
        rf_tile = rf_pulse.unsqueeze(-1) * gains  # (nCh, nSample, nPulse)
        rf_all = rf_tile.reshape(nCh, nSample*nPulse)

        # add small sensor noise
        rf_all = rf_all + 0.002 * rf_all.abs().mean() * torch.randn_like(rf_all)

        # scale to int16
        mx = torch.max(torch.abs(rf_all)) + 1e-9
        scale = (0.8 * 32767.0) / mx
        rf_i16 = torch.clamp(rf_all * scale, -32768, 32767).to(torch.int16)

        # Save in MATLAB-friendly column-major for reshape([L,nCh])
        # We want file layout: channel1 samples contiguous, then channel2, ...
        L = nSample*nPulse
        A = rf_i16.transpose(0, 1).contiguous()          # (L, nCh) but contiguous in row-major
        # Flatten in Fortran order manually: write columns sequentially
        # (equivalent to A.numpy().flatten(order='F') but without CPU reorder surprises)
        A_cpu = A.cpu().numpy()
        data = np.ascontiguousarray(A_cpu.flatten(order="F"))

        fname = os.path.join(OUT_DIR, f"RfData_spc_{b:03d}.bin")
        data.tofile(fname)

        print(f"[{b:03d}/{N_BURST}] wrote {fname}  (scale={float(scale):.3g})")

    print("done.")

if __name__ == "__main__":
    main()
