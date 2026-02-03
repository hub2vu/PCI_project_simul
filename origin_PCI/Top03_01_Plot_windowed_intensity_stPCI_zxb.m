% Written by Sua Bae
%   3/1/2022
%   Modified for simulation data
%
clear; close all;

% set folder
folderidx = 1;  % 시뮬레이션 데이터용 (01_sim)
stFolder = dir(['../Data_tus/' num2str(folderidx,'%02d') '*']);
if isempty(stFolder)
    error('Data_tus 폴더를 찾을 수 없습니다. 경로를 확인하세요: ../Data_tus/%02d*', folderidx);
end
sFolderName = stFolder.name;

% set param
nEig_s = 10;
nEig_e = 90;

sTag = ['_eig' num2str(nEig_s) 'to' num2str(nEig_e)];

% load stPCI_zxb
load([stFolder.folder '/' stFolder.name '/PciData/' 'stPCI_zxb' sTag '.mat'], 'stPCI');
stG = stPCI.stG;
vPCI_zxb = stPCI.vPCI_zxb;

% check image for windowing
figure;
imagesc(stG.aX*1e3, stG.aZ*1e3, db(mean(abs(vPCI_zxb),3),'power'));
hold on;
% set window
% aXlim_m = [-0.15 3.25]*1e-3;
% aZlim_m = [9 12]*1e-3;
aXlim_m = [-2 2]*1e-3 -2e-3;
aZlim_m = [-3 3]*1e-3 +5e-3;
rectangle('Position',[aXlim_m(1)*1e3, aZlim_m(1)*1e3,(aXlim_m(2)-aXlim_m(1))*1e3,(aZlim_m(2)-aZlim_m(1))*1e3]);

[~,xidx_s] = min(abs(stG.aX - aXlim_m(1)));
[~,xidx_e] = min(abs(stG.aX - aXlim_m(2)));
[~,zidx_s] = min(abs(stG.aZ - aZlim_m(1)));
[~,zidx_e] = min(abs(stG.aZ - aZlim_m(2)));

% crop
vPCI_windowed_zxb = vPCI_zxb(zidx_s:zidx_e,xidx_s:xidx_e,:);
% aSigPow = squeeze(mean(reshape(vPCI_windowed_zxb,[numel(vPCI_windowed_zxb)/stG.nPdim, stG.nPdim]),1));
nNumBursts = size(vPCI_windowed_zxb,3);
aSigPow = squeeze(mean(reshape(vPCI_windowed_zxb,[numel(vPCI_windowed_zxb)/nNumBursts, nNumBursts]),1));
% mean(vPCI_windowed(:));

% plot
figure;
plot(aSigPow);
title(sFolderName);
xlabel('burst index');
ylabel('cav sig power of windowed area');

% set(gca, 'YScale', 'log');
% ylim([0.5e6 1e8]);