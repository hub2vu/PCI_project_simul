%% GT_PCI_From_Sources.m
% Ground-truth cavitation activity map (NO transducer, NO beamforming)
% Uses sources.mat (src_x_mm, src_z_mm, src_amp, src_type) and P.mat (stG.aX/aZ grid)
%
% Output:
%   - GT power map (linear)
%   - GT dB map (peak-normalized)
%   - Optional: stable/inertial separate maps if src_type exists

clear; close all; clc;

%% ===== USER PATHS =====
DATA_DIR = fullfile('..','Data_tus','01_sim');   % <-- 너 폴더 구조에 맞게 수정
P_PATH   = fullfile(DATA_DIR, 'P.mat');
S_PATH   = fullfile(DATA_DIR, 'sources.mat');

%% ===== LOAD =====
assert(exist(P_PATH,'file')==2, "P.mat not found: %s", P_PATH);
assert(exist(S_PATH,'file')==2, "sources.mat not found: %s", S_PATH);

load(P_PATH, 'P');       % loads struct P
S = load(S_PATH);        % loads variables from sources.mat

% Required fields
req = {'src_x_mm','src_z_mm','src_amp'};
for i=1:numel(req)
    assert(isfield(S, req{i}), "sources.mat missing field: %s", req{i});
end

stG = P.CAV.stG;

aXmm = stG.aX(:) * 1e3;  % m -> mm
aZmm = stG.aZ(:) * 1e3;  % m -> mm
nx = numel(aXmm);
nz = numel(aZmm);

src_x = single(S.src_x_mm(:));
src_z = single(S.src_z_mm(:));
src_amp = single(S.src_amp(:));

has_type = isfield(S,'src_type');
if has_type
    src_type = uint8(S.src_type(:)); % 0=stable, 1=inertial (네 파이프라인 정의에 맞춤)
else
    src_type = [];
end

fprintf("Loaded %d sources\n", numel(src_amp));
fprintf("Grid: nz=%d, nx=%d\n", nz, nx);
fprintf("X range: %.2f ~ %.2f mm\n", min(aXmm), max(aXmm));
fprintf("Z range: %.2f ~ %.2f mm\n", min(aZmm), max(aZmm));

%% ===== MAP SOURCES TO GRID (NEAREST) =====
% Nearest grid index for each source
ix = interp1(aXmm, 1:nx, double(src_x), 'nearest', 'extrap');
iz = interp1(aZmm, 1:nz, double(src_z), 'nearest', 'extrap');
ix = max(1, min(nx, ix));
iz = max(1, min(nz, iz));

%% ===== BUILD GT MAP (POWER ACCUMULATION) =====
GT = zeros(nz, nx, 'single');           % total power map
GT_st = zeros(nz, nx, 'single');        % stable power (optional)
GT_in = zeros(nz, nx, 'single');        % inertial power (optional)

% "PCI 느낌"으로 power를 누적: amp^2
pwr = src_amp.^2;

for k = 1:numel(pwr)
    GT(iz(k), ix(k)) = GT(iz(k), ix(k)) + pwr(k);

    if has_type
        if src_type(k) == 0
            GT_st(iz(k), ix(k)) = GT_st(iz(k), ix(k)) + pwr(k);
        else
            GT_in(iz(k), ix(k)) = GT_in(iz(k), ix(k)) + pwr(k);
        end
    end
end

%% ===== OPTIONAL: GAUSSIAN SPLAT (SMOOTHER "ACTIVITY") =====
% If you want a smoother GT map that looks like continuous activity, set sigma_mm > 0.
sigma_mm = 0; % e.g., 0.2 or 0.5 (mm). 0 means "no smoothing"
GT_smooth = GT;
if sigma_mm > 0
    % imgaussfilt expects pixels; convert mm sigma to pixel sigma
    dx = (max(aXmm)-min(aXmm)) / max(1,(nx-1));
    dz = (max(aZmm)-min(aZmm)) / max(1,(nz-1));
    sig_px_x = sigma_mm / dx;
    sig_px_z = sigma_mm / dz;
    % MATLAB's imgaussfilt uses isotropic sigma; use imgaussfilt on resized if you need anisotropic.
    % Here we approximate with average pixel sigma:
    sig_px = mean([sig_px_x, sig_px_z]);
    GT_smooth = imgaussfilt(GT, sig_px);
end

%% ===== DISPLAY (dB, peak-normalized) =====
GTdB = 10*log10(GT_smooth + 1e-12);
GTdB = GTdB - max(GTdB(:));     % peak normalize to 0 dB

figure('Name','GT cavitation activity (NO transducer)');
imagesc(aXmm, aZmm, GTdB);
axis image; set(gca,'YDir','reverse');
colormap hot; colorbar;
caxis([-40 0]); % 보기 좋게 조절
xlabel('x (mm)'); ylabel('z (mm)');
title('Ground-truth cavitation activity map (power sum, peak-normalized dB)');

%% ===== OPTIONAL: SHOW STABLE / INERTIAL =====
if has_type
    GTstdB = 10*log10(GT_st + 1e-12); GTstdB = GTstdB - max(GTstdB(:));
    GTindB = 10*log10(GT_in + 1e-12); GTindB = GTindB - max(GTindB(:));

    figure('Name','GT stable vs inertial');
    subplot(1,2,1);
    imagesc(aXmm, aZmm, GTstdB); axis image; set(gca,'YDir','normal');
    colormap hot; colorbar; caxis([-40 0]);
    title('Stable (GT)'); xlabel('x (mm)'); ylabel('z (mm)');

    subplot(1,2,2);
    imagesc(aXmm, aZmm, GTindB); axis image; set(gca,'YDir','normal');
    colormap hot; colorbar; caxis([-40 0]);
    title('Inertial (GT)'); xlabel('x (mm)'); ylabel('z (mm)');
end

%% ===== QUICK SANITY PRINTS =====
fprintf("GT power sum = %.3e\n", sum(GT(:)));
[~, idxMax] = max(GT(:));
[izMax, ixMax] = ind2sub(size(GT), idxMax);
fprintf("GT peak at x=%.3f mm, z=%.3f mm\n", aXmm(ixMax), aZmm(izMax));
