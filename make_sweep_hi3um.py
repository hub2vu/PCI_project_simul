import os, json, time
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

# ================== 입력/출력 경로 ==================
BASE = os.path.dirname(__file__)
NODES_PATH = os.path.join(BASE, "BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_nodes_processed.csv")
EDGES_PATH = os.path.join(BASE, "BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges_processed.csv")

OUT_DIR = os.path.join(BASE, "vessel_sweep_out")
os.makedirs(OUT_DIR, exist_ok=True)

# ================== 설정 (정확도 우선) ==================
VOX_UM = 3.0                 # iso3um -> 1 voxel = 3um (pos_x/y/z는 voxel index로 봄)
OUT_UM = 3.0                 # 출력 격자 해상도(um) 3um 유지
RADIUS_COL = "avgRadiusAvg"  # ✅ 이 값은 "이미 um"로 사용 (절대 *3 하지 말 것)

# --- 코로날 슬라이스(=y 고정 slab) 설정 ---
N_SLICES = 30                # ✅ 요청: 정수리 부근 30장
Y_STEP_POS = 50.0            # pos 단위(=voxel index) 기준 간격 (50이면 150um 간격)
SLAB_DY_POS = 50.0           # slab 두께 (pos 단위). 50이면 150um 두께 (MIP처럼 보이게 하려면 이게 유리)

# --- 스트리밍/성능 ---
CHUNKSIZE = 1_200_000        # edges 스트리밍 청크
BATCH = 6                    # 배치당 슬라이스 수 (메모리 절충)
SAMPLES_PER_PIXEL = 2.0      # centerline 샘플링 밀도 (정확도↑ -> 느려짐↑)

# --- 반지름 안전 상한(비정상 outlier 방지; 큰혈관 보존하려면 충분히 크게) ---
R_PX_CAP = 80                # px cap -> um cap = R_PX_CAP*OUT_UM (여기선 240um)
DO_MORPH = False             # ✅ 두께 왜곡 막기 위해 기본 OFF

# =======================================================

def um_to_px_x(xu, x_min): return (xu - x_min) / OUT_UM
def um_to_px_z(zu, z_min): return (zu - z_min) / OUT_UM

def draw_radius_line(radius_map, x1_px, z1_px, x2_px, z2_px, r_um):
    """
    centerline 픽셀에 radius(um)를 기록.
    겹치면 더 큰 혈관이 우선(최대 유지) -> 큰 혈관이 '크게' 보이는 핵심.
    """
    dx = x2_px - x1_px
    dz = z2_px - z1_px
    L = float(np.hypot(dx, dz))
    n = int(max(2, np.ceil(L * SAMPLES_PER_PIXEL)))
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)

    # 반지름 cap (px->um)
    r_um_cap = R_PX_CAP * OUT_UM
    r_um = float(np.clip(r_um, OUT_UM, r_um_cap))

    H, W = radius_map.shape
    for t in ts:
        cx = int(round(x1_px + t * dx))
        cz = int(round(z1_px + t * dz))
        if 0 <= cz < H and 0 <= cx < W:
            if r_um > radius_map[cz, cx]:
                radius_map[cz, cx] = r_um


print("[1/5] Loading nodes...")
nodes = pd.read_csv(
    NODES_PATH, sep=";", engine="c",
    dtype={"id": np.int64, "pos_x": np.float32, "pos_y": np.float32, "pos_z": np.float32}
)

# pos_* 는 iso3um voxel index로 해석 (정수에 가까움)
x_pos = nodes["pos_x"].to_numpy(np.float32)
y_pos = nodes["pos_y"].to_numpy(np.float32)
z_pos = nodes["pos_z"].to_numpy(np.float32)

# pos -> um (좌표는 voxel*3um이 맞음)
x_um = x_pos * VOX_UM
y_um = y_pos * VOX_UM
z_um = z_pos * VOX_UM

# 출력 grid bounds
pad_um = 0.0
x_min, x_max = float(x_um.min() - pad_um), float(x_um.max() + pad_um)
z_min, z_max = float(z_um.min() - pad_um), float(z_um.max() + pad_um)

W = int(np.ceil((x_max - x_min) / OUT_UM)) + 1
H = int(np.ceil((z_max - z_min) / OUT_UM)) + 1
print(f"Grid (H,W)=({H},{W}), OUT_UM={OUT_UM}")

# ✅ 정수리 부근 30장: y_center는 일단 median 기준으로 잡음(가장 안정적)
y_center_pos = float(np.median(y_pos))
y_start = y_center_pos - (N_SLICES // 2) * Y_STEP_POS
y_list = (y_start + np.arange(N_SLICES, dtype=np.float32) * Y_STEP_POS).astype(np.float32)

print(f"Using y_center_pos={y_center_pos:.1f}")
print(f"Slices: N={N_SLICES}, y_start={y_start:.1f}, step={Y_STEP_POS}, slab_dy={SLAB_DY_POS}")

meta = {
    "VOX_UM": VOX_UM,
    "OUT_UM": OUT_UM,
    "N_SLICES": int(N_SLICES),
    "Y_STEP_POS": float(Y_STEP_POS),
    "SLAB_DY_POS": float(SLAB_DY_POS),
    "y_center_pos": float(y_center_pos),
    "y_list_pos": y_list.tolist(),
    "x_min_um": x_min, "x_max_um": x_max,
    "z_min_um": z_min, "z_max_um": z_max,
    "H": H, "W": W,
    "radius_col": RADIUS_COL,
    "samples_per_pixel": float(SAMPLES_PER_PIXEL),
    "radius_px_cap": int(R_PX_CAP),
    "DO_MORPH": bool(DO_MORPH),
    "notes": "avgRadiusAvg is used as um (no extra scaling). Segment included if its y-range intersects the slab."
}
with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)


# edges 스트리밍 설정
usecols = ["node1id", "node2id", RADIUS_COL]
dtype_edges = {"node1id": np.int64, "node2id": np.int64, RADIUS_COL: np.float32}

print("[2/5] Start coronal sweep (30 slices, hi-accuracy)...")
t_all0 = time.time()

half = SLAB_DY_POS / 2.0

for s0 in range(0, len(y_list), BATCH):
    s1 = min(len(y_list), s0 + BATCH)
    y0s = y_list[s0:s1]
    print(f"\n=== Batch slices {s0}..{s1-1} / {len(y_list)-1} ===")

    # 배치 내 슬라이스별 centerline 반지름 맵(um)
    radius_maps = [np.zeros((H, W), dtype=np.float32) for _ in range(s1 - s0)]

    # edges 파일을 배치당 1회 스캔
    chunk_idx = 0
    for chunk in pd.read_csv(EDGES_PATH, sep=";", engine="c",
                             usecols=usecols, chunksize=CHUNKSIZE, dtype=dtype_edges):
        chunk_idx += 1
        u = chunk["node1id"].to_numpy(np.int64)
        v = chunk["node2id"].to_numpy(np.int64)

        # ✅ 반지름(um) 그대로 사용 (절대 *VOX_UM 하지 말 것)
        r_um = chunk[RADIUS_COL].to_numpy(np.float32)

        # endpoints: x,z in um
        x1 = x_um[u]; z1 = z_um[u]
        x2 = x_um[v]; z2 = z_um[v]

        # endpoints y in pos (for slab intersection)
        y1 = y_pos[u]
        y2 = y_pos[v]
        ylo = np.minimum(y1, y2)
        yhi = np.maximum(y1, y2)

        # px coordinates
        x1_px = um_to_px_x(x1, x_min); z1_px = um_to_px_z(z1, z_min)
        x2_px = um_to_px_x(x2, x_min); z2_px = um_to_px_z(z2, z_min)

        # slice별 slab 교차하는 세그먼트만 선택
        for k, y0 in enumerate(y0s):
            slab_min = float(y0 - half)
            slab_max = float(y0 + half)

            # ✅ 정확한 포함 조건: 세그먼트 y범위가 slab과 교차
            keep = (yhi >= slab_min) & (ylo <= slab_max)
            if not np.any(keep):
                continue

            rm = radius_maps[k]
            for a1, b1, a2, b2, r in zip(x1_px[keep], z1_px[keep], x2_px[keep], z2_px[keep], r_um[keep]):
                draw_radius_line(rm, float(a1), float(b1), float(a2), float(b2), float(r))

        if chunk_idx % 5 == 0:
            print(f"  scanned edge chunk {chunk_idx} ...")

    # [3/5] 각 슬라이스에서 tube 복원 + radius field + dist 저장
    print("[3/5] Build tube + radius field + dist and save...")
    for k, y0 in enumerate(y0s):
        radius_center = radius_maps[k]
        center = (radius_center > 0)

        if not np.any(center):
            out_idx = s0 + k
            np.save(os.path.join(OUT_DIR, f"mask_{out_idx:04d}.npy"), np.zeros((H, W), np.uint8))
            np.save(os.path.join(OUT_DIR, f"dist_um_{out_idx:04d}.npy"), np.full((H, W), 1e6, np.float32))
            np.save(os.path.join(OUT_DIR, f"radius_center_um_{out_idx:04d}.npy"), radius_center)
            np.save(os.path.join(OUT_DIR, f"radius_field_um_{out_idx:04d}.npy"), np.zeros((H, W), np.float32))
            np.save(os.path.join(OUT_DIR, f"vessel_intensity_{out_idx:04d}.npy"), np.zeros((H, W), np.float32))
            print(f"  saved EMPTY slice {out_idx:04d} y0_pos={float(y0):.1f}")
            continue

        # dist to center + nearest center indices -> radius_field 생성
        dist_to_center, (iz, ix) = distance_transform_edt(
            ~center, sampling=(OUT_UM, OUT_UM), return_indices=True
        )
        dist_to_center = dist_to_center.astype(np.float32)

        # 각 픽셀에서 "가장 가까운 centerline"의 반지름(um)
        radius_field = radius_center[iz, ix].astype(np.float32)

        # tube 복원
        vessel_mask = (dist_to_center <= radius_field)

        # 혈관까지 거리맵 (PCI용)
        dist_um = distance_transform_edt(~vessel_mask, sampling=(OUT_UM, OUT_UM)).astype(np.float32)

        # 시각/가중치용 intensity: 큰 혈관 더 밝게 보이도록 radius_field 기반
        # (너가 올린 예시처럼 보이게 하려면 radius_field를 그대로 쓰는 게 가장 직관적)
        vessel_intensity = (radius_field * vessel_mask).astype(np.float32)

        out_idx = s0 + k
        np.save(os.path.join(OUT_DIR, f"mask_{out_idx:04d}.npy"), vessel_mask.astype(np.uint8))
        np.save(os.path.join(OUT_DIR, f"dist_um_{out_idx:04d}.npy"), dist_um)
        np.save(os.path.join(OUT_DIR, f"radius_center_um_{out_idx:04d}.npy"), radius_center)
        np.save(os.path.join(OUT_DIR, f"radius_field_um_{out_idx:04d}.npy"), radius_field)
        np.save(os.path.join(OUT_DIR, f"vessel_intensity_{out_idx:04d}.npy"), vessel_intensity)

        occ = float(vessel_mask.mean())
        rmax = float(radius_field[vessel_mask].max()) if np.any(vessel_mask) else 0.0
        print(f"  saved slice {out_idx:04d} y0_pos={float(y0):.1f} occ={occ:.4f} rmax_um={rmax:.1f}")

t_all1 = time.time()
print(f"\nDONE. Output: {OUT_DIR}")
print(f"Total time: {t_all1 - t_all0:.1f} sec")
