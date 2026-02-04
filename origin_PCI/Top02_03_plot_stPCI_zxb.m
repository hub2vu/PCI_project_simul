% Written by Sua Bae
%   2022-03-17
%       Plot stPCI_zxb in 'PciData' folder
%   Modified for simulation data
%   Updated (2026-02-04): peak-normalized relative dB (0 dB = peak)
%
% NOTE (PCI 관점)
%   기존 코드는 절대 dB + 고정 caxis([45 90])로 플로팅되어,
%   맵 값이 상단에 몰리면 전체가 노랗게 포화되어 보일 수 있습니다.
%   아래 스크립트는 기본적으로 peak 기준 상대 dB(0 dB = peak)를 사용합니다.

clear; close all;

%% ========================= User settings =========================
folderidx  = 1;

% Plot mode
%   'all frames'    : burst/frame별로 반복 플로팅
%   'averaged frame': 3D stack을 평균낸 1장만 플로팅
% sPlotType = 'all frames';
sPlotType  = 'averaged frame';

% Display scale
%   'relative_db' : 0 dB at peak (추천)
%   'absolute_db' : 10log10(power) 그대로 표시
sDisplayScale = 'relative_db';

% Dynamic range for relative dB display (e.g., [-40 0])
relDR_dB = [-40 0];

% Absolute dB caxis (used only when sDisplayScale='absolute_db')
absDR_dB = [45 90];

% If true, prints basic stats (min/max/mean/std) in command window
bPrintStats = true;

%% ========================= Load data =========================
stFolder = dir(['../Data_tus/' num2str(folderidx,'%02d') '*']);
if isempty(stFolder)
    error('Data_tus 폴더를 찾을 수 없습니다. 경로를 확인하세요: ../Data_tus/%02d*', folderidx);
end
sFolderName = stFolder.name;

% Load P
load([stFolder.folder '/' stFolder.name '/P.mat'],'P');

% PciData path
sPciPath = [stFolder.folder '/' stFolder.name '/PciData/'];
if ~exist(sPciPath,'dir'); mkdir(sPciPath); end

% Load stPCI
nEig_s = 10;
nEig_e = 90;
load([sPciPath '/' 'stPCI_zxb' '_eig' num2str(nEig_s) 'to' num2str(nEig_e) '.mat'], 'stPCI');

stG       = stPCI.stG;
vPCI_zxb  = stPCI.vPCI_zxb;   % (z,x,b) power-like map (linear)

%% ========================= Helper functions =========================
% Convert linear power map to dB map
pow2db_safe = @(X) 10*log10(X + eps);

% Peak-normalized relative dB (0 dB = peak)
rel_db = @(X) 10*log10( (X + eps) / (max(X(:)) + eps) );

%% ========================= Plot =========================
if strcmp(sPlotType,'all frames')
    figure;

    for bidx = 1:size(vPCI_zxb,3)
        Pow = vPCI_zxb(:,:,bidx);

        if strcmpi(sDisplayScale,'relative_db')
            Img = rel_db(Pow);
            cax = relDR_dB;
            sUnit = 'dB (rel, peak=0)';
        else
            Img = pow2db_safe(Pow);
            cax = absDR_dB;
            sUnit = 'dB (abs)';
        end

        if bPrintStats
            fprintf('[%s | bidx=%d] min=%.2f, max=%.2f, mean=%.2f, std=%.2f (%s)\n', ...
                sFolderName, bidx, min(Img(:)), max(Img(:)), mean(Img(:)), std(Img(:)), sUnit);
        end

        imagesc(stG.aX*1e3, stG.aZ*1e3, Img);
        axis equal tight; xlabel('x [mm]'); ylabel('z [mm]');
        colorbar; caxis(cax);
        title([sFolderName ' bidx=' num2str(bidx) ' (' num2str(2*(bidx)) 'sec)  [' sUnit ']']);
        drawnow; % pause(0.1);
    end

elseif strcmp(sPlotType,'averaged frame')
    figure;

    Pow = mean(vPCI_zxb,3);

    if strcmpi(sDisplayScale,'relative_db')
        Img = rel_db(Pow);
        cax = relDR_dB;
        sUnit = 'dB (rel, peak=0)';
    else
        Img = pow2db_safe(Pow);
        cax = absDR_dB;
        sUnit = 'dB (abs)';
    end

    if bPrintStats
        fprintf('[%s | averaged] min=%.2f, max=%.2f, mean=%.2f, std=%.2f (%s)\n', ...
            sFolderName, min(Img(:)), max(Img(:)), mean(Img(:)), std(Img(:)), sUnit);
    end

    imagesc(stG.aX*1e3, stG.aZ*1e3, Img);
    axis equal tight; xlabel('x [mm]'); ylabel('z [mm]');
    colorbar; caxis(cax);
    title([sFolderName ': averaged for ' num2str(size(vPCI_zxb,3)) ' bursts  [' sUnit ']']);
    drawnow;

else
    error('Unknown sPlotType: %s', sPlotType);
end
