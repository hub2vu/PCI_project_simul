% Run PCI reconstruction from simulated RfData_spc_*.bin
% - Assumes folder structure:
%   .\Data_tus\01_sim\P.mat
%   .\Data_tus\01_sim\RfData\RfData_spc_001.bin ...
% - Produces:
%   .\Data_tus\01_sim\PciData\stPCI_zxb_eig10to90.mat
%
% IMPORTANT:
%   ExtRun_PCI is designed to be called as ExtRun_PCI(mRfData_spc).
%   Do NOT call ExtRun_PCI with no inputs.

clear; close all; clc;
addpath('src');
global P g
g = gpuDevice();

% ---- session folder (change if needed) ----
sess = fullfile(pwd,'Data_tus','01_sim');
assert(exist(sess,'dir')==7, "Session folder not found: %s", sess);

% ---- load P ----
S = load(fullfile(sess,'P.mat'),'P');
P = S.P;

% ---- PATCH: create P.CAV.mTxDelay_zx_m if missing ----
% Required by InitCuda_PCI.m (line ~58).
% For PCI range-gating, a simple and commonly used default is depth-only delay:
%   mTxDelay_zx_m(z,x) = z (meters) replicated across x.
if ~isfield(P,'CAV') || ~isstruct(P.CAV)
    error('P.CAV missing. Check P.mat');
end
if ~isfield(P.CAV,'stG') || ~isstruct(P.CAV.stG) || ~isfield(P.CAV.stG,'aZ') || ~isfield(P.CAV.stG,'nXdim')
    error('P.CAV.stG (grid) missing. mTxDelay_zx_m cannot be built.');
end
if ~isfield(P.CAV,'mTxDelay_zx_m') || isempty(P.CAV.mTxDelay_zx_m)
    az = single(P.CAV.stG.aZ(:));              % [nZ x 1] in meters
    nx = double(P.CAV.stG.nXdim);
    P.CAV.mTxDelay_zx_m = repmat(az, 1, nx);   % [nZ x nX]
end
% Optional smoothing/offset (matches Sua Bae Top02_01 script) if available:
% if exist('imgaussfilt','file')
%     P.CAV.mTxDelay_zx_m = imgaussfilt(P.CAV.mTxDelay_zx_m,2) - 2e-3;
% end
% ------------------------------------------------------

stRfInfo = P.CAV.stRfInfo;

% offline process flags (match Sua Bae scripts)
P.bProcess = 2;      % 2: offline process
P.CAV.bCompile = 0;
P.bDisplay = 0;      % no display
P.sSessionName = '01_sim';

% eigenspace selection (keep yours)
P.CAV.nEig_s = 10;
P.CAV.nEig_e = 90;

% Initialize GPU/CUDA PCI
InitCuda_PCI();

% list bursts
stList = dir(fullfile(sess,'RfData','RfData_spc_*'));
assert(~isempty(stList), "No RfData_spc_* files under %s", fullfile(sess,'RfData'));

P.CAV.stG.nBdim = numel(stList);

% init PCI (disable plotting to avoid GUI issues)
P.CAV.bPlot = false;
ExtInit_PCI(false);

% allocate output volume that ExtRun_PCI will fill (it uses base/caller scope)
vPCI_zxb = zeros([P.CAV.stG.nZdim, P.CAV.stG.nXdim, P.CAV.stG.nBdim],'single');

% run bursts
for bidx = 1:numel(stList)
    fprintf("[PCI] burst %d / %d : %s\n", bidx, numel(stList), stList(bidx).name);

    % load (time x channel) where time = nSample * nNumPulse
    mRfData_spc = binload(fullfile(stList(bidx).folder, stList(bidx).name), ...
        'int16', [stRfInfo.nSample*P.FUS.nNumPulse, stRfInfo.nChannel]);

    % run PCI for this burst
    ExtRun_PCI(mRfData_spc);
end

% save output
outdir = fullfile(sess,'PciData');
if exist(outdir,'dir')~=7, mkdir(outdir); end

stPCI.stG = P.CAV.stG;
stPCI.vPCI_zxb = vPCI_zxb;
sTag = sprintf('_eig%dto%d', P.CAV.nEig_s, P.CAV.nEig_e);
save(fullfile(outdir, ['stPCI_zxb' sTag '.mat']), 'stPCI','-v7.3');

disp("Done. Saved stPCI at:");
disp(fullfile(outdir, ['stPCI_zxb' sTag '.mat']));
