% Written by Sua
%   Load BfData_zxpo_bidx*_*.bin
%   Summation instead of averaging over foci (1/29/2024)
%   NOTE: 이 스크립트는 BfData가 필요합니다 (시뮬레이션 데이터에는 없음)
%
clear; close all;
addpath('../src');  % origin_PCI 폴더에서 실행하므로 ../src로 수정

folderidx = 1;  % 시뮬레이션 데이터용 (01_sim)
stFolder = dir(['../Data_tus/' num2str(folderidx,'%02d') '*']);
if isempty(stFolder)
    error('Data_tus 폴더를 찾을 수 없습니다. 경로를 확인하세요: ../Data_tus/%02d*', folderidx);
end
sFolderName = stFolder.name;

% Load P
load([stFolder.folder '/' stFolder.name '/P.mat'],'P');
stG = P.CAV.stG;

% Find the number of BfData
stList = dir([stFolder.folder '/' stFolder.name '/BfData/' 'BfData_zxpo_bidx*']);
nNumBfData = numel(stList);

%  PciData path
sPciPath = [stFolder.folder '/' stFolder.name '/PciData/'];
if ~exist(sPciPath,'dir'); mkdir(sPciPath); end

% Parameters
nEig_s = 10; % P.CAV.nEig_s
nEig_e = 90; % P.CAV.nEig_e

figure;
hAx = subplot(1,1,1);
hImgsc = imagesc(hAx, stG.aX*1e3,stG.aZ*1e3,zeros(stG.nZdim,stG.nXdim)); axis(hAx,'equal'); axis(hAx,'tight');
hTitle = title([sFolderName]); colorbar;
clear vPCI_zxob vPCI_zxb
for bidx = 1:numel(stList)    

    % load stPCI
    stPciFile = dir([stFolder.folder '/' stFolder.name '/BfData/' 'stPCI_bidx' num2str(bidx) '_*']);
    load([stPciFile.folder '/' stPciFile.name], 'stPCI');
%     mean(stPCI.vPCI_zxo(:))

    % load BfData
    stBfFile = dir([stFolder.folder '/' stFolder.name '/BfData/' 'BfData_zxpo_bidx' num2str(bidx) '_*']);
    if folderidx == 2
        vBfData_zxpo = binload([stBfFile.folder '/' stBfFile.name],'int16',[stG.nZdim*stG.nXdim,stG.nPdim,stG.nOdim]);
    else
        vBfData_zxpo = binload([stBfFile.folder '/' stBfFile.name],'single',[stG.nZdim*stG.nXdim,stG.nPdim,stG.nOdim]);
    end


    vPCI_zxo = zeros(stG.nZdim,stG.nXdim,stG.nOdim,'single');
    for fociidx = 1:P.FUS.nNumFoci     
        % SVD filtering
        % - set casorati
        mBfData_zxp = squeeze(vBfData_zxpo(:,:,fociidx));  % casorati size = [stG.nZdim*stG.nXdim, stG.nPdim]
        % - decomposition
        [~,~,mV] = svd(ctranspose(mBfData_zxp)*mBfData_zxp,0);
        % - thresholding
        mV_f = mV(:, nEig_s:nEig_e);
        % - reshape
        vFiltered_zxp = reshape(mBfData_zxp*mV_f*ctranspose(mV_f), [stG.nZdim,stG.nXdim,stG.nPdim]);    

        % take absolute and accumulate across pulses (frames)
        vPCI_zxo(:,:,fociidx) = sum(abs(vFiltered_zxp).^2,3);
        
        % plot processed image
        hImgsc.CData = vPCI_zxo(:,:,fociidx);
        hTitle.String = [sFolderName ' bidx = ' num2str(bidx) '/' num2str(numel(stList)) ', fociidx = ' num2str(fociidx) '/' num2str(P.FUS.nNumFoci)];
        drawnow;
    end
    % mPCI_zx = double(mean(vPCI_zxo,3));
    mPCI_zx = double(sum(vPCI_zxo,3));
    
    vPCI_zxob(:,:,:,bidx) = vPCI_zxo;
    vPCI_zxb(:,:,bidx) = mPCI_zx;
    
end
    
% SAVE stPCI_zxb in 'PciData' folder
stPCI.stG       = stPCI.stG;
stPCI.vPCI_zxob = vPCI_zxob;
stPCI.vPCI_zxb 	= vPCI_zxb;
stPCI.nEig_s    = nEig_s;
stPCI.nEig_e    = nEig_e;

save([sPciPath '/' 'stPCI_zxb' '_eig' num2str(stPCI.nEig_s) 'to' num2str(stPCI.nEig_e) '.mat'], 'stPCI'); 
