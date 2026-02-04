% Written by Sua Bae
%   3/1/2022
%   Modified for simulation data
%
clear; close all;

% Data_tus/01_sim 경로 직접 지정
sessPath = '../Data_tus/01_sim';
if ~exist(sessPath, 'dir')
    error('데이터 폴더를 찾을 수 없습니다: %s', sessPath);
end
sFolderName = '01_sim';

% set param
nEig_s = 10;
nEig_e = 90;

sTag = ['_eig' num2str(nEig_s) 'to' num2str(nEig_e)];

% load stPCI_zxb
load(fullfile(sessPath, 'PciData', ['stPCI_zxb' sTag '.mat']), 'stPCI');
stG = stPCI.stG;
vPCI_zxb = stPCI.vPCI_zxb;

% set entities for making video
figure('color','w');
sVideoName = fullfile(sessPath, 'PciData', ['PCI' sTag '_test']);
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