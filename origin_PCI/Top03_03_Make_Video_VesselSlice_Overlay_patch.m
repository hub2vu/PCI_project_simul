% Top03_03_Make_Video_VesselSlice_Overlay_patch.m
% Modified for simulation data - Uses vessel slice image instead of ULM
% Original by Sua Bae (3/1/2022)
%
% 뇌 혈관 슬라이스 이미지와 PCI를 오버레이하여 비디오 생성
%
clear; close all;

% ====== 경로 설정 ======
addpath('../src');

%% ================ 설정 ================
SLICE_IDX = 14;
dataFolder = '../Data_tus/01_sim';
sFolderName = '01_sim';
sVesselDir = '../vessel_sweep_out';

if ~exist(dataFolder, 'dir')
    error('데이터 폴더를 찾을 수 없습니다: %s', dataFolder);
end

%% ================ 혈관 슬라이스 로드 ================
disp('Loading vessel slice data...');

% 혈관 이미지 로드
sVesselFile = fullfile(sVesselDir, sprintf('vessel_intensity_%04d.npy', SLICE_IDX));
if ~exist(sVesselFile, 'file')
    error('혈관 슬라이스 파일을 찾을 수 없습니다: %s', sVesselFile);
end

mVessel = readNPY(sVesselFile);
mVessel = double(mVessel);
[nZ_vessel, nX_vessel] = size(mVessel);
disp(['Vessel slice loaded: ', num2str(nZ_vessel), ' x ', num2str(nX_vessel)]);

%% ================ PCI 데이터 로드 ================
disp('Loading PCI data...');
nPci_Eig_s = 10;
nPci_Eig_e = 90;
sPciPath = fullfile(dataFolder, 'PciData');
pciFile = fullfile(sPciPath, ['stPCI_zxb_eig' num2str(nPci_Eig_s) 'to' num2str(nPci_Eig_e) '.mat']);

if ~exist(pciFile, 'file')
    error('PCI 파일을 찾을 수 없습니다: %s', pciFile);
end
load(pciFile, 'stPCI');

vPCI = double(stPCI.vPCI_zxb);
stG_pci = stPCI.stG;
stG_pci.aX = double(stG_pci.aX);
stG_pci.aZ = double(stG_pci.aZ);

if max(abs(stG_pci.aX)) > 1.0 % mm to m conversion
    stG_pci.aX = stG_pci.aX * 1e-3;
    stG_pci.aZ = stG_pci.aZ * 1e-3;
end

%% ================ Vessel 좌표계 재정의 & 보간 ================
stG_vessel.aX = linspace(min(stG_pci.aX), max(stG_pci.aX), nX_vessel);
stG_vessel.aZ = linspace(min(stG_pci.aZ), max(stG_pci.aZ), nZ_vessel);

disp('Interpolating images...');
nPixelSize_m = 30e-6; 
[mVessel_itp, stG_vessel_itp] = interpImg(mVessel, stG_vessel, nPixelSize_m);

clear vPCI_itp_zxb
for bidx = 1:size(vPCI,3)
    [vPCI_itp_zxb(:,:,bidx), stG_pci_itp] = interpImg(vPCI(:,:,bidx), stG_pci, nPixelSize_m);
end

aFOV = [min(stG_pci.aX)*1e3, max(stG_pci.aX)*1e3, min(stG_pci.aZ)*1e3, max(stG_pci.aZ)*1e3];

%% ================ 비디오 생성 (Peak-normalized dB Patch) ================
bVideo = 1;
if bVideo
    sVideoDir = fullfile(dataFolder, 'PciData');
    if ~isfolder(sVideoDir); mkdir(sVideoDir); end
    sVideoName = fullfile(sVideoDir, ['PCI_VesselOverlay_slice' num2str(SLICE_IDX,'%04d')]);
    
    vidObj = VideoWriter(sVideoName, 'MPEG-4');
    vidObj.FrameRate = 5;
    open(vidObj);
    
    fig = figure('Position', [100 100 800 600], 'color', 'w');
    hAx1 = subplot(1,1,1);
    hAx2 = axes('Position', hAx1.Position);
    linkaxes([hAx1, hAx2]);
    
    % 배경: 혈관
    imagesc(hAx1, stG_vessel_itp.aX*1e3, stG_vessel_itp.aZ*1e3, mVessel_itp);
    colormap(hAx1, 'gray');
    caxis(hAx1, [0, prctile(mVessel_itp(:), 99.5)]);
    set(hAx1, 'YDir', 'reverse');
    
    % 오버레이: PCI 초기화
    hPCI = imagesc(hAx2, stG_pci_itp.aX*1e3, stG_pci_itp.aZ*1e3, zeros(stG_pci_itp.nZdim, stG_pci_itp.nXdim));
    colormap(hAx2, 'parula');
    set(hAx2, 'YDir', 'reverse', 'Visible', 'off');
    
    % Patch 설정
    bPeakNormDb = true; 
    DR_DB = 40;
    SHOW_PEAK_MARKER = true;
    
    if SHOW_PEAK_MARKER
        hold(hAx1, 'on');
        hPk = plot(hAx1, nan, nan, 'r.', 'MarkerSize', 20);
    end

    hCb = colorbar(hAx2);
    hCb.Label.String = 'PCI Intensity [dB rel. peak]';
    
    vPCI_lin = max(vPCI_itp_zxb, 0);
    nNumBursts = size(vPCI_lin, 3);

    for bidx = 1:nNumBursts
        mLin = vPCI_lin(:,:,bidx);
        pkVal = max(mLin(:));
        
        if pkVal > 0
            mDb = 10 * log10(mLin ./ pkVal + 1e-12);
        else
            mDb = ones(size(mLin)) * -DR_DB;
        end
        
        % 데이터 업데이트
        hPCI.CData = mDb;
        caxis(hAx2, [-DR_DB, 0]);
        
        % AlphaData: DR 범위에 맞춰 가시성 조절
        A = (mDb - (-DR_DB)) / DR_DB;
        hPCI.AlphaData = min(max(A, 0), 1);
        
        % Peak Marker 업데이트
        if SHOW_PEAK_MARKER && pkVal > 0
            [~, maxIdx] = max(mDb(:));
            [iz, ix] = ind2sub(size(mDb), maxIdx);
            hPk.XData = stG_pci_itp.aX(ix)*1e3;
            hPk.YData = stG_pci_itp.aZ(iz)*1e3;
        end
        
        title(hAx1, sprintf('%s - Burst %d/%d', sFolderName, bidx, nNumBursts));
        axis(hAx1, aFOV); axis(hAx2, aFOV);
        drawnow;
        
        writeVideo(vidObj, getframe(fig));
    end
    close(vidObj);
    disp(['Video saved: ' sVideoName]);
end

%% ================ Helper Function: readNPY (Robust Shape Parsing) ================
function data = readNPY(filename)
    fid = fopen(filename, 'r');
    if fid == -1, error('Cannot open file: %s', filename); end
    
    magic = fread(fid, 6, 'uint8');
    version = fread(fid, 2, 'uint8');
    if version(1) == 1
        headerLen = fread(fid, 1, 'uint16');
    else
        headerLen = fread(fid, 1, 'uint32');
    end
    
    headerStr = char(fread(fid, headerLen, 'char')'); % 변수명 통일: headerStr
    
    % dtype 추출
    dtypeMatch = regexp(headerStr, '''descr''\s*:\s*''([^'']+)''', 'tokens', 'once');
    if isempty(dtypeMatch)
        dtypeMatch = regexp(headerStr, '"descr"\s*:\s*"([^"]+)"', 'tokens', 'once');
    end
    dtypeStr = dtypeMatch{1};
    
    % ---------------------------------------------------------
    % Parse shape (ROBUST)
    % ---------------------------------------------------------
    if isempty(headerStr)
        error('readNPY:HeaderEmpty', 'NPY header string is empty. Cannot parse shape.');
    end

    % Try regex patterns commonly seen in .npy headers
    shapeStr = '';
    tok = regexp(headerStr, 'shape''\s*:\s*\(([^)]*)\)', 'tokens', 'once'); % e.g. "shape': (256, 256)"
    if isempty(tok)
        tok = regexp(headerStr, 'shape"\s*:\s*\(([^)]*)\)', 'tokens', 'once'); % in case of double quotes
    end
    if isempty(tok)
        tok = regexp(headerStr, 'shape\s*:\s*\(([^)]*)\)', 'tokens', 'once'); % more permissive
    end
    if ~isempty(tok)
        shapeStr = strtrim(tok{1});
    end

    if isempty(shapeStr)
        % As a fallback, try to locate "shape" and grab between '(' and ')'
        iShape = strfind(headerStr, 'shape');
        if ~isempty(iShape)
            i0 = strfind(headerStr(iShape(1):end), '(');
            i1 = strfind(headerStr(iShape(1):end), ')');
            if ~isempty(i0) && ~isempty(i1) && i1(1) > i0(1)
                shapeStr = strtrim(headerStr(iShape(1)+i0(1) : iShape(1)+i1(1)-2));
            end
        end
    end

    if isempty(shapeStr)
        error('readNPY:ShapeParseFail', ...
            "Failed to parse 'shape' from NPY header. Header was:\n%s", headerStr);
    end

    % Convert "256, 256," -> [256 256]
    shapeStr = strrep(shapeStr, ',', ' ');
    shape = str2num(['[' shapeStr ']']); %#ok<ST2NM>
    
    if isempty(shape)
        % 1차원 배열인 경우 (예: "256,") 처리
        shape = 1; 
    end
    
    if dtypeStr(1) == '<' || dtypeStr(1) == '>' || dtypeStr(1) == '|'
        dtypeStr = dtypeStr(2:end);
    end
    
    switch dtypeStr
        case 'f4', matlabType = 'single';
        case 'f8', matlabType = 'double';
        case 'i2', matlabType = 'int16';
        case 'i4', matlabType = 'int32';
        case 'u1', matlabType = 'uint8';
        otherwise, error('Unsupported dtype: %s', dtypeStr);
    end
    
    data = fread(fid, prod(shape), matlabType);
    fclose(fid);
    
    if numel(shape) > 1
        data = reshape(data, fliplr(shape));
        data = permute(data, ndims(data):-1:1);
    end
end