% Written by Sua (01/29/2024)
%   Load and Plot BfData_zxpo_bidx*_*.bin
%   NOTE: 이 스크립트는 BfData가 필요합니다 (시뮬레이션 데이터에는 없음)
%
clear; close all;
addpath('../src');

% Data_tus/01_sim 경로 직접 지정
sessPath = '../Data_tus/01_sim';
if ~exist(sessPath, 'dir')
    error('데이터 폴더를 찾을 수 없습니다: %s', sessPath);
end
sFolderName = '01_sim';

% Load P
load(fullfile(sessPath, 'P.mat'), 'P');
stG = P.CAV.stG;

% Find the number of BfData
stList = dir(fullfile(sessPath, 'BfData', 'BfData_zxpo_bidx*'));
nNumBfData = numel(stList);

figure;
hAx = subplot(1,1,1);
hImgsc = imagesc(hAx, stG.aX*1e3,stG.aZ*1e3,zeros(stG.nZdim,stG.nXdim)); axis(hAx,'equal'); axis(hAx,'tight');
hTitle = title([sFolderName]); colorbar;
clear vPCI_zxob vPCI_zxb
for bidx = 1:numel(stList)       

    % load BfData
    stBfFile = dir([stFolder.folder '/' stFolder.name '/BfData/' 'BfData_zxpo_bidx' num2str(bidx) '_*']);
    vBfData_zxpo = binload([stBfFile.folder '/' stBfFile.name],'int16',[stG.nZdim*stG.nXdim,stG.nPdim,stG.nOdim]);
    %     vBfData_zxpo = binload([stBfFile.folder '/' stBfFile.name],'single',[stG.nZdim*stG.nXdim,stG.nPdim,stG.nOdim]);


    vPCI_zxo = zeros(stG.nZdim,stG.nXdim,stG.nOdim,'single');
    for fociidx = 1:P.FUS.nNumFoci     
        mBfData_zxp = squeeze(vBfData_zxpo(:,:,fociidx));  % casorati size = [stG.nZdim*stG.nXdim, stG.nPdim]

        % reshape
        vBfData_zxp = reshape(mBfData_zxp,[stG.nZdim,stG.nXdim,stG.nPdim]);


    pidx = 50; % pulse index out of 100

        imagesc(squeeze(vBfData_zxp(:,:,pidx)));
        title([sFolderName ' bidx = ' num2str(bidx) '/' num2str(numel(stList)) ', fociidx = ' num2str(fociidx) '/' num2str(P.FUS.nNumFoci)]);
        drawnow;
    end
    
end