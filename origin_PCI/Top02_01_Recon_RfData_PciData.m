% Written by Sua Bae
%   2022-03-01
%       Load RfData_*.bin files and beamforming
%   Modified for simulation data
%
clear; close all;
addpath('../src');  % origin_PCI 폴더에서 실행하므로 ../src로 수정
global P g
g = gpuDevice();

% set folder
folderidx = 1;
stFolder = dir(['../Data_tus/' num2str(folderidx,'%02d') '*']);
if isempty(stFolder)
    error('Data_tus 폴더를 찾을 수 없습니다. 경로를 확인하세요: ../Data_tus/%02d*', folderidx);
end
sFolderName = stFolder.name;


% Load P
load([stFolder.folder '/' stFolder.name '/P']);
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
stList = dir([stFolder.folder '/' stFolder.name '/RfData/' 'RfData_spc_*']);
nNumBurst = numel(stList);
P.CAV.stG.nBdim = nNumBurst;

% Run external initialize function for PCI
ExtInit_PCI();

% loop for all bursts
vPCI_zxb = zeros([P.CAV.stG.nZdim, P.CAV.stG.nXdim, P.CAV.stG.nBdim]);
for bidx = 1:numel(stList)
    % load RfData
    mRfData_spc = binload([stList(bidx).folder '/' stList(bidx).name], 'int16', [stRfInfo.nSample*P.FUS.nNumPulse, stRfInfo.nChannel]);
    
    % Run external processing function for PCI
    ExtRun_PCI(mRfData_spc); % it will stack each mPCI_zx into vPCI_zxb
        
end

mkdir([stFolder.folder '/' stFolder.name '/PciData/']);

stPCI.stG = P.CAV.stG;
stPCI.vPCI_zxb = vPCI_zxb;
sTag = ['_eig' num2str(P.CAV.nEig_s) 'to' num2str(P.CAV.nEig_e)];
save([stFolder.folder '/' stFolder.name '/PciData/' 'stPCI_zxb' sTag '.mat'], 'stPCI');


% 
% figure;
% aIntensity = squeeze(mean(mean(vPCI_zxb,1),2));
% plot(aIntensity);
% xlabel('bidx'); ylabel('mean intensity of PCI');