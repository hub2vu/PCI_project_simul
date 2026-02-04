% RunPCI_01_sim_v2.m
% 
% P.mat 설정을 그대로 사용하는 PCI reconstruction.
% - mTxDelay_zx_m 패치 제거 (원래 P.mat 값 사용)
% - make_sources_from_slice_v2.py + gen_rf_cuda_v3.py와 일관된 설정
%
% 폴더 구조 (Data_tus/01_sim 기준):
%   ./Data_tus/01_sim/P.mat
%   ./Data_tus/01_sim/sources.mat
%   ./Data_tus/01_sim/RfData/RfData_spc_001.bin ...
%   ./Data_tus/01_sim/PciData/  (출력)

clear; close all; clc;
addpath('src');
global P g
g = gpuDevice();

% ---- Session folder ----
% 모든 데이터는 Data_tus/01_sim 경로에서 처리
sess = fullfile(pwd, 'Data_tus', '01_sim');
if ~exist(sess, 'dir')
    error('세션 폴더가 없습니다: %s\n먼저 gen_rf_cuda_v3.py를 실행하세요.', sess);
end
fprintf('Session folder: %s\n', sess);

% ---- Load P.mat ----
fprintf('Loading P.mat...\n');
S = load(fullfile(sess, 'P.mat'), 'P');
P = S.P;

% ---- Verify P.CAV.mTxDelay_zx_m exists ----
% 원래 P.mat에 이미 있어야 함. 없으면 에러.
if ~isfield(P.CAV, 'mTxDelay_zx_m') || isempty(P.CAV.mTxDelay_zx_m)
    % 만약 없다면 vTxDelay_zxo_m의 첫 번째 foci 사용
    if isfield(P.CAV, 'vTxDelay_zxo_m') && ~isempty(P.CAV.vTxDelay_zxo_m)
        warning('mTxDelay_zx_m not found. Using vTxDelay_zxo_m(:,:,1) as fallback.');
        P.CAV.mTxDelay_zx_m = P.CAV.vTxDelay_zxo_m(:,:,1);
    else
        error('P.CAV.mTxDelay_zx_m is missing and no fallback available.');
    end
end

fprintf('mTxDelay_zx_m size: [%d, %d]\n', size(P.CAV.mTxDelay_zx_m, 1), size(P.CAV.mTxDelay_zx_m, 2));
fprintf('mTxDelay range: %.3f ~ %.3f mm\n', ...
    min(P.CAV.mTxDelay_zx_m(:))*1000, max(P.CAV.mTxDelay_zx_m(:))*1000);

stRfInfo = P.CAV.stRfInfo;

% ---- Process flags ----
P.bProcess = 2;         % 2: offline
P.CAV.bCompile = 0;
P.bDisplay = 0;
P.sSessionName = '01_sim';

% ---- Eigenspace selection ----
P.CAV.nEig_s = 10;
P.CAV.nEig_e = 90;

% ---- Initialize CUDA PCI ----
fprintf('\nInitializing CUDA PCI...\n');
InitCuda_PCI();

% ---- List RF files ----
rfDir = fullfile(sess, 'RfData');
stList = dir(fullfile(rfDir, 'RfData_spc_*.bin'));
assert(~isempty(stList), 'No RfData_spc_* files found in %s', rfDir);
fprintf('Found %d RF files.\n', numel(stList));

P.CAV.stG.nBdim = numel(stList);

% ---- Initialize PCI (no plotting) ----
P.CAV.bPlot = false;
ExtInit_PCI(false);

% ---- Allocate output ----
vPCI_zxb = zeros([P.CAV.stG.nZdim, P.CAV.stG.nXdim, P.CAV.stG.nBdim], 'single');

% ---- Process each burst ----
fprintf('\nProcessing bursts...\n');
for bidx = 1:numel(stList)
    fprintf('[PCI] burst %d / %d : %s\n', bidx, numel(stList), stList(bidx).name);
    
    % Load RF data: (time x channel)
    % time = nSample * nNumPulse
    mRfData_spc = binload(fullfile(stList(bidx).folder, stList(bidx).name), ...
        'int16', [stRfInfo.nSample * P.FUS.nNumPulse, stRfInfo.nChannel]);
    
    % Run PCI
    ExtRun_PCI(mRfData_spc);
end

% ---- Save output ----
outDir = fullfile(sess, 'PciData');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

stPCI.stG = P.CAV.stG;
stPCI.vPCI_zxb = vPCI_zxb;
sTag = sprintf('_eig%dto%d', P.CAV.nEig_s, P.CAV.nEig_e);
outFile = fullfile(outDir, ['stPCI_zxb' sTag '.mat']);
save(outFile, 'stPCI', '-v7.3');

fprintf('\n=== Done ===\n');
fprintf('Saved: %s\n', outFile);

% ---- Quick visualization ----
figure;
imagesc(P.CAV.stG.aX*1000, P.CAV.stG.aZ*1000, sum(vPCI_zxb, 3));
axis image;
colormap hot;
colorbar;
xlabel('x (mm)');
ylabel('z (mm)');
title(sprintf('Total PCI (eig %d-%d)', P.CAV.nEig_s, P.CAV.nEig_e));
