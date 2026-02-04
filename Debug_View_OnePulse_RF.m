%% Debug_View_OnePulse_RF.m
clear; close all; clc;

ROOT = uigetdir(pwd, 'Select ...\Data_tus\01_sim folder');
if isequal(ROOT,0), error('Canceled'); end

load(fullfile(ROOT,'P.mat'),'P');
nSample = double(P.CAV.stRfInfo.nSample);
nCh     = double(P.CAV.stRfInfo.nChannel);
nPulse  = double(P.FUS.nNumPulse);

RF_DIR = fullfile(ROOT,'RfData');
L = dir(fullfile(RF_DIR,'RfData_spc_*.bin'));
assert(~isempty(L), "No bin in %s", RF_DIR);

[~,ix] = max([L.datenum]);
binPath = fullfile(L(ix).folder, L(ix).name);
fprintf("Reading: %s\n", binPath);

fid = fopen(binPath,'rb');
raw = fread(fid, nSample*nPulse*nCh, 'int16=>double');
fclose(fid);

A = reshape(raw, [nSample*nPulse, nCh]); % (L, ch)
rf = A.';                                % (ch, L)

% ---- pulse 하나만 떼서 보기 ----
p = 1; % 1~nPulse 중 보고 싶은 펄스
idx = (p-1)*nSample + (1:nSample);
rf1 = rf(:, idx);

figure;
imagesc(rf1);
colormap gray; colorbar;
xlabel('sample (within 1 pulse)'); ylabel('channel');
title(sprintf('RF of pulse %d (ch x nSample)', p));
set(gca,'YDir','normal');
