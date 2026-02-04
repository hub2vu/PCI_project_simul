% Written by Sua Bae
%   2022-03-17
%       + Save stPCI_zxb in 'PciData' folder
%   2022-03-01
%       Load stPCI and plot
%   NOTE: 이 스크립트는 BfData 폴더의 실시간 stPCI 파일이 필요합니다 (시뮬레이션에서는 사용 불가)
%   
clear;% close all;

% Data_tus/01_sim 경로 직접 지정
sessPath = '../Data_tus/01_sim';
if ~exist(sessPath, 'dir')
    error('데이터 폴더를 찾을 수 없습니다: %s', sessPath);
end
sFolderName = '01_sim';

% Load P
load(fullfile(sessPath, 'P.mat'), 'P');

% Get a list of stPCI
stList = dir(fullfile(sessPath, 'BfData', 'stPCI_*'));

%  PciData path
sPciPath = fullfile(sessPath, 'PciData');
if ~exist(sPciPath,'dir'); mkdir(sPciPath); end

figure;
hAx = subplot(1,1,1);
clear vPCI_zxob vPCI_zxb
for bidx = 1:numel(stList)
    
    % load stPCI
    stPciFile = dir([stFolder.folder '/' stFolder.name '/BfData/' 'stPCI_bidx' num2str(bidx) '_*']);
    load([stPciFile.folder '/' stPciFile.name], 'stPCI');
    
    % stack
    vPCI_zxob(:,:,:,bidx) = stPCI.vPCI_zxo;
    vPCI_zxb(:,:,bidx) = stPCI.mPCI_zx;
    
    % plot
    imagesc(stPCI.stG.aX*1e3, stPCI.stG.aZ*1e3, 10*log10(stPCI.mPCI_zx)); 
    axis equal tight; xlabel('[mm]'); ylabel('[mm]');
    colorbar; caxis([45 90]);
    title([sFolderName ' bidx=' num2str(bidx) ' (' num2str(2*(bidx)) 'sec)']);
    drawnow; %pause(0.1);
end

figure;
aIntensity = squeeze(mean(mean(vPCI_zxb,1),2));
plot(aIntensity);
xlabel('bidx'); ylabel('mean intensity of PCI');
title([sFolderName]);
ylim([0 5e7]);


% SAVE stPCI_zxb in 'PciData' folder
stG = stPCI.stG; % real-time-recon pci
clear stPCI; % clear "real-time-recon" pci
stPCI.stG       = stG;
stPCI.vPCI_zxob = vPCI_zxob;
stPCI.vPCI_zxb 	= vPCI_zxb;
stPCI.nEig_s    = P.CAV.nEig_s;
stPCI.nEig_e    = P.CAV.nEig_e;

save([sPciPath '/' 'stPCI_zxb' '_eig' num2str(stPCI.nEig_s) 'to' num2str(stPCI.nEig_e) '.mat'], 'stPCI'); 
