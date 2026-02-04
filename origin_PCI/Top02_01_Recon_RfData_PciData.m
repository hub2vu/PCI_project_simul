% Written by Sua Bae
%   2022-03-01
%       Load RfData_*.bin files and beamforming
%   Modified for simulation data
%
% 사용법:
%   1. 프로젝트 루트에서 직접 실행: RunPCI_01_sim_v2.m 사용 권장
%   2. origin_PCI 폴더에서 실행: 아래 DATA_ROOT 경로 설정 필요
%
clear; close all;

% ====== 경로 설정 ======
% 스크립트 위치 기준 경로 자동 탐지
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);  % origin_PCI의 상위 폴더 = 프로젝트 루트

addpath(fullfile(projectRoot, 'src'));

global P g
g = gpuDevice();

% ====== 데이터 경로 설정 ======
% Option 1: 프로젝트 루트 구조 (P.mat이 루트에 있는 경우)
% Option 2: Data_tus 구조 (기존 구조)
DATA_ROOT = '';  % 빈 문자열이면 프로젝트 루트 사용

if isempty(DATA_ROOT)
    % 프로젝트 루트에서 데이터 찾기
    dataFolder = projectRoot;
    sFolderName = 'sim';

    % P.mat 확인
    if ~exist(fullfile(dataFolder, 'P.mat'), 'file')
        error('P.mat을 찾을 수 없습니다: %s', fullfile(dataFolder, 'P.mat'));
    end
else
    % Data_tus 구조 사용
    folderidx = 1;
    stFolder = dir(fullfile(DATA_ROOT, [num2str(folderidx,'%02d') '*']));
    if isempty(stFolder)
        error('Data 폴더를 찾을 수 없습니다: %s/%02d*', DATA_ROOT, folderidx);
    end
    dataFolder = fullfile(stFolder.folder, stFolder.name);
    sFolderName = stFolder.name;
end

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

%%%%%%%%
% PATCH: P.CAV.mTxDelay_zx_m이 없으면 생성 (시뮬레이션 데이터용)
if ~isfield(P.CAV,'mTxDelay_zx_m') || isempty(P.CAV.mTxDelay_zx_m)
    az = single(P.CAV.stG.aZ(:));              % [nZ x 1] in meters
    nx = double(P.CAV.stG.nXdim);
    P.CAV.mTxDelay_zx_m = repmat(az, 1, nx);   % [nZ x nX]
    fprintf('[PATCH] P.CAV.mTxDelay_zx_m 생성됨 (depth-only delay)\n');
end

P.CAV.mTxDelay_zx_m = imgaussfilt(P.CAV.mTxDelay_zx_m,2) - 2e-3;
% P.CAV.mTxDelay_zx_m = single(repmat((P.CAV.stG.aZ + 2e-3)',[1,P.CAV.stG.nXdim]));
% P.CAV.mTxDelay_zx_m = single(repmat((P.CAV.stG.aZ)',[1,P.CAV.stG.nXdim]));
% figure;imagesc(P.CAV.stG.aX*1e3,P.CAV.stG.aZ*1e3,P.CAV.mTxDelay_zx_m*1e3); axis equal tight; colorbar; title('');
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
vPCI_zxb = zeros([P.CAV.stG.nZdim, P.CAV.stG.nXdim, P.CAV.stG.nBdim]);
for bidx = 1:numel(stList)
    % load RfData
    rfFile = fullfile(stList(bidx).folder, stList(bidx).name);
    mRfData_spc = binload(rfFile, 'int16', [stRfInfo.nSample*P.FUS.nNumPulse, stRfInfo.nChannel]);

    % Run external processing function for PCI
    ExtRun_PCI(mRfData_spc); % it will stack each mPCI_zx into vPCI_zxb

end

% Output directory
pciDataDir = fullfile(dataFolder, 'PciData');
if ~exist(pciDataDir, 'dir')
    mkdir(pciDataDir);
end

stPCI.stG = P.CAV.stG;
stPCI.vPCI_zxb = vPCI_zxb;
sTag = ['_eig' num2str(P.CAV.nEig_s) 'to' num2str(P.CAV.nEig_e)];
save(fullfile(pciDataDir, ['stPCI_zxb' sTag '.mat']), 'stPCI');


% 
% figure;
% aIntensity = squeeze(mean(mean(vPCI_zxb,1),2));
% plot(aIntensity);
% xlabel('bidx'); ylabel('mean intensity of PCI');