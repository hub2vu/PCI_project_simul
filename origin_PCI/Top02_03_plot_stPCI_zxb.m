% Written by Sua Bae
%   2022-03-17
%       Plot stPCI_zxb in 'PciData' folder
%   Modified for simulation data
%
clear; close all;

folderidx = 1;
stFolder = dir(['../Data_tus/' num2str(folderidx,'%02d') '*']);
if isempty(stFolder)
    error('Data_tus 폴더를 찾을 수 없습니다. 경로를 확인하세요: ../Data_tus/%02d*', folderidx);
end
sFolderName = stFolder.name;

% Load P
load([stFolder.folder '/' stFolder.name '/P.mat'],'P');

%  PciData path
sPciPath = [stFolder.folder '/' stFolder.name '/PciData/'];
if ~exist(sPciPath,'dir'); mkdir(sPciPath); end

% Load
nEig_s = 10;
nEig_e = 90;
load([sPciPath '/' 'stPCI_zxb' '_eig' num2str(nEig_s) 'to' num2str(nEig_e) '.mat'], 'stPCI'); 

stG       = stPCI.stG;
vPCI_zxb = stPCI.vPCI_zxb;

% Plot
% sPlotType = 'all frames';
sPlotType = 'averaged frame';

if strcmp(sPlotType,'all frames')
    figure;
    for bidx = 1:size(vPCI_zxb,3)
        imagesc(stG.aX*1e3, stG.aZ*1e3, 10*log10(vPCI_zxb(:,:,bidx))); 
        axis equal tight; xlabel('x [mm]'); ylabel('z [mm]');
        colorbar; caxis([45 90]);
        title([sFolderName ' bidx=' num2str(bidx) ' (' num2str(2*(bidx)) 'sec)']);
        drawnow; %pause(0.1);
    end
elseif strcmp(sPlotType,'averaged frame')
    figure;
    imagesc(stG.aX*1e3, stG.aZ*1e3, 10*log10(mean(vPCI_zxb,3))); 
    axis equal tight; xlabel('x [mm]'); ylabel('z [mm]');
    colorbar; caxis([45 90]);
    title([sFolderName ': averaged for ' num2str(size(vPCI_zxb,3)) ' bursts']);
    drawnow; %pause(0.1);
end
