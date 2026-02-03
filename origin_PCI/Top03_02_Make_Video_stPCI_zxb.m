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

% set entities for making video
figure('color','w');
sVideoName = [stFolder.folder '/' stFolder.name '/PciData/' 'PCI' sTag '_test'];
disp(['Creating... ' sVideoName '.mp4']);
vidObj = VideoWriter(sVideoName,'MPEG-4');
vidObj.FrameRate = 10;
open(vidObj);
hImg = imagesc(stG.aX*1e3, stG.aZ*1e3, zeros(stG.nZdim,stG.nXdim));
colorbar; axis equal tight;
hTitle = title('');
xlabel('[mm]'); ylabel('[mm]');
% loop
nMax = max(vPCI_zxb(:));
for bidx = 1:size(vPCI_zxb,3)
    
    % log compression
    mPCI_zx = vPCI_zxb(:,:,bidx);
%     mPCI_zx_db = 10*log10(abs(mPCI_zx/max(mPCI_zx(:)))); % normalized by each burst
    mPCI_zx_db = 10*log10(abs(mPCI_zx/nMax)); % normalized by entire burst
    
    % plot
    hImg.CData = mPCI_zx_db;
    hTitle.String = [sFolderName ', bidx=' num2str(bidx) ' (' num2str(2*bidx) 'sec) ' strrep(sTag,'_',' ')];
    caxis([-20 0]);
    axis([-4.5 4.5 2.5 8]);
    
    currFrame = getframe(gcf);
    writeVideo(vidObj,currFrame);
end
    
close(vidObj);


% figure;imagesc(stG.aX*1e3, stG.aZ*1e3, P.CAV.mTxDelay_zx_m*1e3);
% colorbar; axis equal tight;
% hTitle = title('');
% xlabel('[mm]'); ylabel('[mm]');
% % loop
% hTitle.String = 'mTxDelay (mm)'