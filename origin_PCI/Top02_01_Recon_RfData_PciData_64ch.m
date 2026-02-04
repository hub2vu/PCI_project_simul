%% 
% Written by Sua Bae
%   2022-03-01
%       Load RfData_*.bin files and beamforming
%   Modified for simulation data (Data_tus/01_sim)
%
clear; close all;

% ====== 경로 설정 ======
addpath('../src');

global P g
g = gpuDevice();
USE_64CH_ODD = true;   % true: 1,3,5,...,127 채널 사용 / false: 원본 128ch 그대로

% Data_tus/01_sim 경로 직접 지정
dataFolder = '../Data_tus/01_sim';
sFolderName = '01_sim';

if ~exist(dataFolder, 'dir')
    error('데이터 폴더를 찾을 수 없습니다: %s', dataFolder);
end

% P.mat 확인
if ~exist(fullfile(dataFolder, 'P.mat'), 'file')
    error('P.mat을 찾을 수 없습니다: %s', fullfile(dataFolder, 'P.mat'));
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
% ====== PATCH: 128ch 파일을 읽되, 처리용으로는 64ch만 쓰기 ======
%  - 파일 저장 채널수(nChFile)는 유지
%  - PCI 처리용 채널수(nChProc) 및 트랜스듀서 채널 관련 벡터는 축소
nChFile = double(stRfInfo.nChannel);   % bin 파일에 들어있는 채널 수(보통 128)
chPick  = 1:nChFile;
if USE_64CH_ODD
    chPick = 1:2:nChFile;             % odd channels: 1,3,5,...,127
    fprintf('[PATCH] USE_64CH_ODD=ON -> using %d/%d channels (odd indices)\n', numel(chPick), nChFile);

    % Update processing-side stRfInfo
    stRfInfo_proc = stRfInfo;
    stRfInfo_proc.nChannel = int32(numel(chPick));
    P.CAV.stRfInfo = stRfInfo_proc;
    stRfInfo = stRfInfo_proc;         % 이후 코드는 처리용 64ch 기준

    % Update transducer geometry vectors if they match original channel count
    oldN = nChFile;
    fn = fieldnames(stTrans);
    for k = 1:numel(fn)
        v = stTrans.(fn{k});
        try
            if isnumeric(v) && isvector(v) && numel(v)==oldN
                stTrans.(fn{k}) = v(chPick);
            end
        catch
        end
    end
    P.stTrans = stTrans;
end
%%%%%%%%
% PATCH: P.CAV.mTxDelay_zx_m이 없으면 생성 (시뮬레이션 데이터용)
if ~isfield(P.CAV,'mTxDelay_zx_m') || isempty(P.CAV.mTxDelay_zx_m)
    P.CAV.mTxDelay_zx_m = zeros(P.CAV.stG.nZdim, P.CAV.stG.nXdim, 'single');
    fprintf('[PATCH] P.CAV.mTxDelay_zx_m set to ZERO (passive/point-source)\n');
end

%P.CAV.mTxDelay_zx_m = imgaussfilt(P.CAV.mTxDelay_zx_m,2) - 2e-3;
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
    nSample = double(stRfInfo.nSample);
    nChRead = nChFile;
    nChProc = double(stRfInfo.nChannel);
    nPulse  = double(P.FUS.nNumPulse);
    L       = nSample * nPulse;

    fid = fopen(rfFile,'rb');
    assert(fid>0, "Cannot open RF bin: %s", rfFile);
    raw = fread(fid, L*nChRead, 'int16=>double');
    fclose(fid);
    assert(numel(raw)==L*nChRead, "RF bin size mismatch. got=%d expected=%d (%s)", numel(raw), L*nChRead, rfFile);
    rf_chL     = reshape(raw, [nChRead, L]);  % (ch, L)  channel-major
    if USE_64CH_ODD
        rf_chL = rf_chL(chPick, :);           % (64, L)
        assert(size(rf_chL,1)==nChProc, "Channel pick mismatch: picked=%d stRfInfo.nChannel=%d", size(rf_chL,1), nChProc);
    end
    mRfData_spc = rf_chL.';               % (L, ch)  keep ExtRun_PCI interface
 
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