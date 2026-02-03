% Edited by Sua Bae (8/24/2022)
% Written by Sua Bae (3/1/2022)
%   NOTE: 이 스크립트는 ULM 데이터(Data_flw)가 필요합니다
%
clear; close all;
addpath('../src');  % origin_PCI 폴더에서 실행하므로 ../src로 수정
% %% Focus info
% mFocus_xz_m =[-1.65e-3, 5e-3; % ele81 = 1.65e-3
%               -1.75e-3, 5e-3; % ele82 = 1.75e-3
%               -1.85e-3, 5e-3;
%               -1.95e-3, 5e-3;
%               -2.05e-3, 5e-3;]; % [m] size = P.FUS.nNumFoci x 2(lateral, axial), positions of foci  <- THIS IS FUS DEPTH ************
% aFocalSize_xz = [0.1 8]*1e-3; % 0.1 mm  x 8 mm


%% ULM

% ULM folder 
folderidx_ulm = 13; % M02: 13, M03: 25
stFolder = dir(['../Data_flw/' num2str(folderidx_ulm,'%02d') ' *']);
sUlmPath = [stFolder.folder '\' stFolder.name '\ULM\'];

sMouse = stFolder.name(4:6);
disp(['sMouse : ' sMouse]);

sUlm_GridDensity = 'x2'; 
nUlm_Eig_s = 30;
nUlm_Eig_e = 300;  

stFolder = dir(['../Data_flw/' num2str(folderidx_ulm,'%02d') ' *']);
sBfPath = [stFolder.folder '\' stFolder.name '/BfData_' sUlm_GridDensity '/']; 

% load parameter
load([sBfPath 'stParam'], 'stParam');
stRfInfo = stParam.stRfInfo;

% choose data
if strcmp(sMouse,'M02')
    sFileName = [sUlmPath 'stULM_' sUlm_GridDensity '_eig' num2str(nUlm_Eig_s) 'to' num2str(nUlm_Eig_e) ...
                          '_th1000_intp2'];
elseif strcmp(sMouse,'M03')
    sFileName = [sUlmPath 'stULM_' sUlm_GridDensity '_eig' num2str(nUlm_Eig_s) 'to' num2str(nUlm_Eig_e) ...
                          '_th3000_intp2x1'];
end

% load data
disp(['Loading...' sFileName '.mat']);
load([sFileName '.mat'], 'stULM');

% plot
didx_s = 1;
didx_e = size(stULM.vMap,3);

aDataIndices = zeros(1,size(stULM.vMap,3));
aDataIndices(didx_s:didx_e) = 1;
%     aDataIndices([52 56 67 82]) = 0;
aDataIndices = logical(aDataIndices);

% get ULM
mULM_org = mean(stULM.vMap(:,:,aDataIndices),3);

% Notch filtering to remove grid artifact
mULM = rmGridArtifact(mULM_org, 2, 1, 0);

% figure;
%     imagesc(stULM.stG.aX*1e3, stULM.stG.aZ*1e3, mULM);colorbar; axis equal tight;  colormap(hot);
%     title([stFolder.name ': didx=' num2str(didx_s) ':' num2str(didx_e) ...
%            ', eig=' num2str(nUlm_Eig_s) 'to' num2str(nUlm_Eig_e) ...
%            ', th=' num2str(nUlm_Threshold) ', intp=' num2str(nUlm_Intp)]);
%     ylabel('mm'); xlabel('mm');
%     caxis([0.5 30]);
%    axis([-4.5 4.5 2.5 7.9]);
   

%% Parameters per mouse

%- Field of view
if strcmp(sMouse,'M02');        aFOV = [-4.5 4.5 2.5 7];
elseif strcmp(sMouse,'M03');    aFOV = [-4.5 4.5 2.6 7.1];
end


%% TUS (Passive cavitation imaging)

%  PciData path
stFolder = dir(['../Data_tus/*' sMouse '*']);
sFolderName = stFolder.name;

% Load
nPci_Eig_s = 10;
nPci_Eig_e = 90;
load([stFolder.folder '/' stFolder.name '/PciData/' 'stPCI_zxb' '_eig' num2str(nPci_Eig_s) 'to' num2str(nPci_Eig_e) '.mat'], 'stPCI'); 

vPCI = stPCI.vPCI_zxb;

% figure;
%     imagesc(stPCI.stG.aX*1e3, stPCI.stG.aZ*1e3, vPCI(:,:,30)); 
%     axis equal tight; xlabel('x [mm]'); ylabel('z [mm]');
%     colorbar; caxis([45 90]);
%     title([sFolderName ': averaged for ' num2str(size(vPCI_zxb,3)) ' bursts']);
%     drawnow; %pause(0.1);
%     axis([-4.5 4.5 2.5 7.9]);


%% Interpolation to enhance the resolution of pictures..
%% + nOffset_m

disp('Interpolating image...');

% - ULM
nPixelSize_m = 10e-6;
[mULM_itp,    stG_ulm_itp] = interpImg(mULM,    stULM.stG, nPixelSize_m);
% - PCI
clear vPCI_itp
for bidx = 1:size(vPCI,3)
    [vPCI_itp_zxb(:,:,bidx), stG_pci_itp] = interpImg(vPCI(:,:,bidx), stPCI.stG, nPixelSize_m);
end

% Total PCI Intensity
mPCI_itp_zx = sum(vPCI_itp_zxb,3);

% Offset between FLW and PCI
nDepthOffset_m = 0e-3;

%% Ploting the Total PCI Intensity  (Figure 3A)

% Figure Object
figure('Position', [30 331 780 480], 'color','w');
hAx1 = subplot(1,1,1);        
hAx2 = axes('Position',hAx1.Position);
linkaxes([hAx1, hAx2]); 
% - ULM
hAx1_imagesc = imagesc(hAx1, stG_ulm_itp.aX*1e3, stG_ulm_itp.aZ*1e3, mULM_itp); 
colormap(hAx1,'gray');
if strcmp(sMouse,'M02')
    caxis(hAx1,[0 20]);
elseif strcmp(sMouse,'M03')
    caxis(hAx1,[0 22]);
else
    caxis(hAx1,[0 20]);
end


% - PCI
bLog = 1;
if bLog
    mPCI_final = log10(mPCI_itp_zx+1e-10);
    if strcmp(sMouse,'M02')
        nDR = [9 11]; % 10^nDR
    elseif strcmp(sMouse,'M03')
        nDR = [9 12]; % 10^nDR
    else
        nDR = [8.5 11.5]; % 10^nDR
    end   
else
    mPCI_final = max(mPCI_itp_zx,0);
    nDR = [0 5e10];
end
hAx2_imagesc = imagesc(hAx2, stG_pci_itp.aX*1e3, (stG_pci_itp.aZ + nDepthOffset_m)*1e3, mPCI_final); 
hAx2_imagesc.AlphaData = (max(hAx2_imagesc.CData,nDR(1))-nDR(1))/(nDR(2)-nDR(1));
colormap(hAx2,'parula');
caxis(hAx2, nDR);


% - Axes setting
hAx1.Color = 'k';
hAx2.Visible = 'off';
axis(hAx1,'equal'); axis(hAx1,'tight');
axis(hAx2,'equal'); axis(hAx2,'tight');
xlabel(hAx1,'x (mm)'); ylabel(hAx1,'z (mm)');

% axis(hAx1,[stG_ulm_itp.aX(1) stG_ulm_itp.aX(end) stG_ulm_itp.aZ(1) stG_ulm_itp.aZ(end)]*1e3);
% axis(hAx1,[-3.5 3.5 2.5 7]);
% axis(hAx1,[-4.5 4.5 2.5 7]);

hAx1_cb = colorbar(hAx1);
hAx2_cb = colorbar(hAx2);
hAx1_cb.Visible = 0;
hAx2_cb.Label.String = 'Total PCI intensity';
hTitle = title(hAx1, [stFolder.name]);
hAx1.FontSize = 15;
hAx2.FontSize = 15;
axis(hAx1,aFOV); axis(hAx2,aFOV);

if bLog
    hAx2_cb.Ticks = nDR(1):1:nDR(2);
    for tkidx = 1:numel(hAx2_cb.Ticks); hAx2_cb.TickLabels{tkidx} = ['10^{' num2str(hAx2_cb.Ticks(tkidx)) '}']; end
end
    

%% Writing a video (Movie S1)

bVideo = 1;

if bVideo
    % Video Object
    if ~isfolder('video'); mkdir('video'); end
    sVideoName = ['video/' '_PCI_' sMouse '_eig' num2str(nPci_Eig_s) 'to' num2str(nPci_Eig_e) '_ULM_eig' num2str(nUlm_Eig_s) 'to' num2str(nUlm_Eig_e)];
    disp(['Creating... ' sVideoName '.mp4']);
    vidObj = VideoWriter(sVideoName,'MPEG-4');
    vidObj.FrameRate = 10;
    open(vidObj);

    % Figure Object
    figure('Position', [800 331 780 480], 'color','w');
    hAx1 = subplot(1,1,1);        
    hAx2 = axes('Position',hAx1.Position);
    linkaxes([hAx1, hAx2]); 
    % - ULM
    hAx1_imagesc = imagesc(hAx1, stG_ulm_itp.aX*1e3, stG_ulm_itp.aZ*1e3, mULM_itp); 
    colormap(hAx1,'gray');
    if strcmp(sMouse,'M02')
        caxis(hAx1,[0 20]);
    elseif strcmp(sMouse,'M03')
        caxis(hAx1,[0 22]);
    else
        caxis(hAx1,[0 20]);
    end
    % - PCI
    hAx2_imagesc = imagesc(hAx2, stG_pci_itp.aX*1e3, (stG_pci_itp.aZ + nDepthOffset_m)*1e3, zeros(stG_pci_itp.nXdim,stG_pci_itp.nZdim)); 
    hAx2_imagesc.AlphaData = (max(hAx2_imagesc.CData,nDR(1))-nDR(1))/(nDR(2)-nDR(1));
    colormap(hAx2,'parula');
    caxis(hAx2, nDR);
    % - Axes setting
    hAx1.Color = 'k';
    hAx2.Visible = 'off';
    axis(hAx1,'equal'); axis(hAx1,'tight');
    axis(hAx2,'equal'); axis(hAx2,'tight');
    xlabel(hAx1,'x (mm)'); ylabel(hAx1,'z (mm)');
    hAx1_cb = colorbar(hAx1);
    hAx2_cb = colorbar(hAx2);
    hAx1_cb.Visible = 0;
    hAx2_cb.Label.String = 'PCI intensity';
    hTitle = title(hAx1, '');
    hAx1.FontSize = 15;
    hAx2.FontSize = 15;


    % - PCI dynamic range
    bLog = 1;
    if bLog
        vPCI_final = log10(vPCI_itp_zxb+1e-10);
        if strcmp(sMouse,'M02')
            nDR = [7 9.5]; % 10^nDR
        elseif strcmp(sMouse,'M03')
            nDR = [7 9.5]+0.5; % 10^nDR
        else
            nDR = [7 9.5]; % 10^nDR
        end  
    else
        vPCI_final = max(vPCI_itp_zxb,0);
        nDR = [0 5e10];
    end

    % - Time offset!
    %   Since the microbubble injected later, we need to set the starting time
    %   point so that the video has the frames only after the bubble injection    
    if strcmp(sMouse,'M02')
        nStartTime_s = 18;
    elseif strcmp(sMouse,'M03')
        nStartTime_s = 0;
    else
        nStartTime_s = 20;
    end  
    nEndTime_s = nStartTime_s + 120; % for 2 min
    bidx_s = nStartTime_s/2 + 1; % PRF: 0.5 Hz
    bidx_e = nEndTime_s/2; % PRF: 0.5 Hz

    % loop
    for bidx = bidx_s:bidx_e % size(vPCI_final,3)

        % update PCI frame
        hAx2_imagesc.CData = vPCI_final(:,:,bidx);
        hAx2_imagesc.AlphaData = (max(hAx2_imagesc.CData,nDR(1))-nDR(1))/(nDR(2)-nDR(1));
        % update title
        % hTitle.String = [sFolderName ', bidx = ' num2str(bidx) ' (t = ' num2str(2*bidx) ' s)'];
        hTitle.String = ['t = ' num2str(2*(bidx-bidx_s+1)) ' s'];
        caxis(hAx2, nDR);
        if bLog
            hAx2_cb.Ticks = nDR(1):0.5:nDR(2);
            for tkidx = 1:numel(hAx2_cb.Ticks); hAx2_cb.TickLabels{tkidx} = ['10^{' num2str(hAx2_cb.Ticks(tkidx)) '}']; end
        end
        axis(hAx1,aFOV); axis(hAx2,aFOV);

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
end