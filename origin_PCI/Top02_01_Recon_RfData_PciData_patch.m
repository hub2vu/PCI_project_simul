%%
% Written by Sua Bae
%    2022-03-01
%        Load RfData_*.bin files and beamforming
%    Modified for simulation data (Data_tus/01_sim)
%
clear; close all;

% ====== 경로 설정 ======
scriptDir   = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
addpath(fullfile(projectRoot, 'src'));

global P g vPCI_zxb

% ====== GPU 선택 (없으면 CPU로 안전 폴백) ======
try
    g = gpuDevice();
catch
    g = [];
    fprintf('[INFO] No GPU device available -> running on CPU\n');
end

% ====== 데이터 경로 자동 탐지 ======
% 1) origin_PCI 폴더에서 실행: ../Data_tus/01_sim
% 2) 프로젝트 루트에서 실행: ./Data_tus/01_sim
candidate1 = fullfile(projectRoot, 'Data_tus', '01_sim');
candidate2 = fullfile(scriptDir,   '..', 'Data_tus', '01_sim');
if exist(candidate1, 'dir')
    dataFolder = candidate1;
elseif exist(candidate2, 'dir')
    dataFolder = candidate2;
else
    error('Data_tus/01_sim 폴더를 찾을 수 없습니다.\n  tried: %s\n         %s', candidate1, candidate2);
end
sFolderName = '01_sim';

% P.mat 확인
assert(exist(fullfile(dataFolder, 'P.mat'), 'file')==2, 'P.mat을 찾을 수 없습니다: %s', fullfile(dataFolder, 'P.mat'));

% Load P
load(fullfile(dataFolder, 'P.mat'), 'P');
stRfInfo = P.CAV.stRfInfo;
stTrans = P.stTrans;

P.bProcess = 2; % 2: offline process
P.CAV.bCompile = 0;
P.bDisplay = 1;
P.sSessionName = sFolderName;

P.CAV.nEig_s = 10;
P.CAV.nEig_e = 90;

% ====== (선택) 후처리/표시/누적 파라미터 ======
% ExtRun_PCI 내부 알고리즘을 건드리기 어려운 대신,
% 저장/표시에서 쓸 수 있는 표준 파라미터를 P.CAV.post로 통일해 둡니다.
P.CAV.post.bDbNorm         = true;   % true면 dB로도 저장
P.CAV.post.dbFloor         = -40;    % 표시 dynamic range 하한 (dB)
P.CAV.post.bBurstNormalize = true;   % burst별 max로 정규화 후 평균 (display용)
P.CAV.post.eps             = 1e-12;

%%%%%%%%
% PATCH: P.CAV.mTxDelay_zx_m이 없으면 생성 (시뮬레이션 데이터용)
if ~isfield(P.CAV,'mTxDelay_zx_m') || isempty(P.CAV.mTxDelay_zx_m)
    % PASSIVE/point-source 가정: TX delay를 쓰면 깊이-게이팅이 망가질 수 있어 0으로 둡니다.
    P.CAV.mTxDelay_zx_m = zeros(P.CAV.stG.nZdim, P.CAV.stG.nXdim, 'single');
    fprintf('[PATCH] P.CAV.mTxDelay_zx_m set to ZERO (passive/point-source)\n');
end

%P.CAV.mTxDelay_zx_m = imgaussfilt(P.CAV.mTxDelay_zx_m,2) - 2e-3;
%%%%%%%%%%%

% Initialize GPU
InitCuda_PCI();

% Get a list of RfData
rfDataDir = fullfile(dataFolder, 'RfData');
stList = dir(fullfile(rfDataDir, 'RfData_spc_*'));
nNumBurst = numel(stList);
P.CAV.stG.nBdim = nNumBurst;

if nNumBurst == 0
    error('RfData 파일을 찾을 수 없습니다: %s', rfDataDir);
end
fprintf('Found %d RF data files.\n', nNumBurst);

% Run external initialize function for PCI
ExtInit_PCI();

% loop for all bursts
% NOTE: ExtRun_PCI가 global vPCI_zxb에 누적하는 구현인 경우가 많아서
%       반드시 global로 두고 초기화합니다.
vPCI_zxb = zeros([P.CAV.stG.nZdim, P.CAV.stG.nXdim, P.CAV.stG.nBdim], 'single');
for bidx = 1:numel(stList)
    % load RfData (gen_rf_cuda_v3.py 출력 포맷: channel-major)
    rfFile  = fullfile(stList(bidx).folder, stList(bidx).name);
    nSample = double(stRfInfo.nSample);
    nCh     = double(stRfInfo.nChannel);
    nPulse  = double(P.FUS.nNumPulse);
    L       = nSample * nPulse;

    fid = fopen(rfFile,'rb');
    assert(fid>0, "Cannot open RF bin: %s", rfFile);
    raw = fread(fid, L*nCh, 'int16=>single');
    fclose(fid);
    assert(numel(raw)==L*nCh, "RF bin size mismatch. got=%d expected=%d (%s)", numel(raw), L*nCh, rfFile);

    % raw 저장 순서:
    %   Python에서 rf[ch, t] 형태로 flatten 했으면 MATLAB에서는 [nCh, L]로 reshape가 맞습니다.
    rf_chL      = reshape(raw, [nCh, L]);  % (ch, L)
    mRfData_spc = rf_chL.';                % (L, ch)  ExtRun_PCI 인터페이스 유지

    % Run external processing function for PCI
    try
        if nargout('ExtRun_PCI') >= 1
            mPCI_zx = ExtRun_PCI(mRfData_spc);
            if ~isempty(mPCI_zx)
                vPCI_zxb(:,:,bidx) = gather(single(mPCI_zx));
            end
        else
            ExtRun_PCI(mRfData_spc);
        end
    catch ME
        fprintf('[WARN] ExtRun_PCI call failed at bidx=%d (%s)\n', bidx, rfFile);
        rethrow(ME);
    end
end

% Output directory
pciDataDir = fullfile(dataFolder, 'PciData');
if ~exist(pciDataDir, 'dir')
    mkdir(pciDataDir);
end

stPCI.stG = P.CAV.stG;
stPCI.vPCI_zxb = vPCI_zxb;
stPCI.post = P.CAV.post;

% ====== 누적/평균 맵 (display/평가용) ======
% (1) 단순 평균 (linear power)
stPCI.mPCI_zx_mean = mean(vPCI_zxb, 3);

% (2) burst별 peak 정규화 후 평균 (깊이에 따른 전반적 gain 편차를 줄여 확인용)
if isfield(P.CAV,'post') && isfield(P.CAV.post,'bBurstNormalize') && P.CAV.post.bBurstNormalize
    v = vPCI_zxb;
    mx_b = squeeze(max(max(v,[],1),[],2));            % [nB x 1]
    mx_b = reshape(max(mx_b, P.CAV.post.eps), 1, 1, []);
    v_norm = v ./ mx_b;
    stPCI.mPCI_zx_mean_norm = mean(v_norm, 3);
end

% (선택) dB 버전도 같이 저장 (표시/디버그 용)
if isfield(P.CAV,'post') && isfield(P.CAV.post,'bDbNorm') && P.CAV.post.bDbNorm
    mx = max(vPCI_zxb(:));
    stPCI.vPCI_zxb_db = 10*log10( max(vPCI_zxb, P.CAV.post.eps) / max(mx, P.CAV.post.eps) );

    mxm = max(stPCI.mPCI_zx_mean(:));
    stPCI.mPCI_zx_mean_db = 10*log10( max(stPCI.mPCI_zx_mean, P.CAV.post.eps) / max(mxm, P.CAV.post.eps) );
    if isfield(stPCI,'mPCI_zx_mean_norm')
        mxn = max(stPCI.mPCI_zx_mean_norm(:));
        stPCI.mPCI_zx_mean_norm_db = 10*log10( max(stPCI.mPCI_zx_mean_norm, P.CAV.post.eps) / max(mxn, P.CAV.post.eps) );
    end
end

sTag = ['_eig' num2str(P.CAV.nEig_s) 'to' num2str(P.CAV.nEig_e)];
save(fullfile(pciDataDir, ['stPCI_zxb' sTag '.mat']), 'stPCI');

fprintf('PCI Reconstruction Complete. Saved to: %s\n', pciDataDir);