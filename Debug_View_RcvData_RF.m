%% Debug_View_RfBin_RF.m
% 목적: gen_rf_cuda_v3.py가 만든 RfData_spc_###.bin을 읽어서
%       raw RF(채널 x 샘플) 이미지로 확인 (하이퍼볼라 체크)

clear; close all; clc;

ROOT = uigetdir(pwd, 'Select ...\Data_tus\01_sim folder');
if isequal(ROOT,0), error('Canceled'); end

RF_DIR = fullfile(ROOT, 'RfData');
assert(exist(RF_DIR,'dir')==7, "RfData folder not found: %s", RF_DIR);

% 가장 최신 bin 선택
L = dir(fullfile(RF_DIR, '*.bin'));
assert(~isempty(L), "No .bin under RfData: %s", RF_DIR);

[~,ix] = max([L.datenum]);
binPath = fullfile(L(ix).folder, L(ix).name);
fprintf("Loading BIN: %s\n", binPath);

% P.mat에서 nSample/nChannel/nNumPulse 읽기 (정확한 reshape 위해)
load(fullfile(ROOT,'P.mat'),'P');
stRfInfo = P.CAV.stRfInfo;
nSample = double(stRfInfo.nSample);
nCh     = double(stRfInfo.nChannel);
nPulse  = double(P.FUS.nNumPulse);
Ltot    = nSample * nPulse;

fprintf("Expect Ltot=nSample*nPulse = %d * %d = %d\n", nSample, nPulse, Ltot);
fprintf("Expect nCh=%d\n", nCh);

% BIN 읽기: int16, MATLAB은 column-major이므로 그대로 reshape 하면 됨
fid = fopen(binPath, 'rb');
assert(fid>0, "Failed to open: %s", binPath);
raw = fread(fid, Ltot*nCh, 'int16=>double');
fclose(fid);

assert(numel(raw)==Ltot*nCh, "File size mismatch. Read %d, expected %d", numel(raw), Ltot*nCh);

A = reshape(raw, [Ltot, nCh]);     % (L, nCh)
rf_ch_ts = A.';                    % (nCh, L)

% 보기 좋게 정규화해서 표시
scale = prctile(abs(rf_ch_ts(:)), 99.9) + 1e-12;

figure('Name','Raw RF from BIN (channel x time)');
imagesc(rf_ch_ts/scale);
colormap gray; colorbar;
xlabel('sample (L = nSample*nPulse)'); ylabel('channel');
title('Raw RF (normalized). Point source -> hyperbola should appear.');
set(gca,'YDir','normal');

% 중간 채널 파형
mid = round(nCh/2);
figure('Name','Mid channel RF (from BIN)');
plot(rf_ch_ts(mid,:)); grid on;
xlabel('sample'); ylabel('amplitude');
title(sprintf('Mid-channel RF (ch=%d)', mid));
