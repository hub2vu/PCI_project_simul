% Top03_03_Make_Video_VesselSlice_Overlay.m
% Modified for simulation data - Uses vessel slice image instead of ULM
% Original by Sua Bae (3/1/2022)
%
% 뇌 혈관 슬라이스 이미지와 PCI를 오버레이하여 비디오 생성
% make_sources_from_slice.py에서 사용한 SLICE_IDX와 동일한 슬라이스 사용
%
clear; close all;

% ====== 경로 설정 ======
addpath('../src');

%% ================ 설정 ================
% 시뮬레이션에 사용된 슬라이스 인덱스 (make_sources_from_slice_v2.py의 SLICE_IDX와 일치)
SLICE_IDX = 14;

% Data_tus/01_sim 경로 직접 지정
dataFolder = '../Data_tus/01_sim';
sFolderName = '01_sim';
sVesselDir = '../vessel_sweep_out';

if ~exist(dataFolder, 'dir')
    error('데이터 폴더를 찾을 수 없습니다: %s', dataFolder);
end


%% ================ 혈관 슬라이스 로드 ================
disp('Loading vessel slice data...');

% meta.json 로드
fid = fopen(fullfile(sVesselDir, 'meta.json'), 'r');
raw = fread(fid, inf);
fclose(fid);
str = char(raw');
meta = jsondecode(str);

% 혈관 이미지 로드 (vessel_intensity 사용 - 큰 혈관이 더 밝게 보임)
sVesselFile = fullfile(sVesselDir, sprintf('vessel_intensity_%04d.npy', SLICE_IDX));
if ~exist(sVesselFile, 'file')
    error('혈관 슬라이스 파일을 찾을 수 없습니다: %s', sVesselFile);
end

% Python npy 파일 로드 (MATLAB에서 직접 로드)
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

vPCI = stPCI.vPCI_zxb;
vPCI = double(vPCI);
stG_pci = stPCI.stG;
% [추가] 좌표축 double 변환 및 단위 보정 (mm -> m)
stG_pci.aX = double(stG_pci.aX);
stG_pci.aZ = double(stG_pci.aZ);

% 만약 PCI 좌표가 mm 단위라면(값이 1보다 크다면) m 단위로 변환
if max(abs(stG_pci.aX)) > 1.0
    disp('Warning: stG_pci appears to be in mm. Converting to meters.');
    stG_pci.aX = stG_pci.aX * 1e-3;
    stG_pci.aZ = stG_pci.aZ * 1e-3;
end
disp(['PCI loaded: ', num2str(size(vPCI,1)), ' x ', num2str(size(vPCI,2)), ' x ', num2str(size(vPCI,3)), ' bursts']);
disp(['PCI X range: ', num2str(stG_pci.aX(1)*1e3), ' ~ ', num2str(stG_pci.aX(end)*1e3), ' mm']);
disp(['PCI Z range: ', num2str(stG_pci.aZ(1)*1e3), ' ~ ', num2str(stG_pci.aZ(end)*1e3), ' mm']);
%% ================ (핵심) Vessel 좌표계를 PCI 좌표계에 맞춰 재정의 ================
% Vessel 이미지는 픽셀 단위(3um)로 생성된 단면이지만,
% overlay에서는 "같은 좌표계"로 놓여야 함.
% -> stG_pci의 (aX/aZ) 범위에 vessel 이미지가 선형으로 매핑되도록 정의.

aX_pci_m = stG_pci.aX(:);
aZ_pci_m = stG_pci.aZ(:);

stG_vessel.aX = linspace(min(aX_pci_m), max(aX_pci_m), nX_vessel);
stG_vessel.aZ = linspace(min(aZ_pci_m), max(aZ_pci_m), nZ_vessel);
stG_vessel.nXdim = nX_vessel;
stG_vessel.nZdim = nZ_vessel;

aX_vessel_mm = stG_vessel.aX * 1e3;
aZ_vessel_mm = stG_vessel.aZ * 1e3;
disp('Vessel grid re-mapped to PCI coordinate system:');
disp(['Vessel X range: ', num2str(aX_vessel_mm(1)), ' ~ ', num2str(aX_vessel_mm(end)), ' mm']);
disp(['Vessel Z range: ', num2str(aZ_vessel_mm(1)), ' ~ ', num2str(aZ_vessel_mm(end)), ' mm']);

%% ================ 이미지 보간 (해상도 맞추기) ================
disp('Interpolating images...');

nPixelSize_m = 30e-6;  % 30um 해상도로 보간 (속도와 품질 균형)

% 혈관 이미지 보간
[mVessel_itp, stG_vessel_itp] = interpImg(mVessel, stG_vessel, nPixelSize_m);

% PCI 이미지 보간
clear vPCI_itp_zxb
for bidx = 1:size(vPCI,3)
    [vPCI_itp_zxb(:,:,bidx), stG_pci_itp] = interpImg(vPCI(:,:,bidx), stG_pci, nPixelSize_m);
end

% Total PCI Intensity
mPCI_itp_total = sum(vPCI_itp_zxb, 3);

%% ================ FOV 설정 (두 이미지의 공통 영역) ================
% PCI와 Vessel 이미지의 겹치는 영역 계산
aX_pci_mm = stG_pci.aX * 1e3;
aZ_pci_mm = stG_pci.aZ * 1e3;

% 공통 X 범위
x_common_min = max(aX_vessel_mm(1), aX_pci_mm(1));
x_common_max = min(aX_vessel_mm(end), aX_pci_mm(end));

% 공통 Z 범위
z_common_min = max(aZ_vessel_mm(1), aZ_pci_mm(1));
z_common_max = min(aZ_vessel_mm(end), aZ_pci_mm(end));

aFOV = [x_common_min, x_common_max, z_common_min, z_common_max];

disp('');
disp('=== 좌표계 비교 ===');
disp(['PCI X: ', num2str(aX_pci_mm(1), '%.2f'), ' ~ ', num2str(aX_pci_mm(end), '%.2f'), ' mm']);
disp(['Vessel X: ', num2str(aX_vessel_mm(1), '%.2f'), ' ~ ', num2str(aX_vessel_mm(end), '%.2f'), ' mm']);
disp(['PCI Z: ', num2str(aZ_pci_mm(1), '%.2f'), ' ~ ', num2str(aZ_pci_mm(end), '%.2f'), ' mm']);
disp(['Vessel Z (with offset): ', num2str(aZ_vessel_mm(1), '%.2f'), ' ~ ', num2str(aZ_vessel_mm(end), '%.2f'), ' mm']);
disp(['Common FOV: X=[', num2str(aFOV(1), '%.2f'), ', ', num2str(aFOV(2), '%.2f'), '], Z=[', num2str(aFOV(3), '%.2f'), ', ', num2str(aFOV(4), '%.2f'), '] mm']);

% 공통 영역이 없으면 경고
if x_common_min >= x_common_max || z_common_min >= z_common_max
    warning('PCI와 Vessel 이미지의 겹치는 영역이 없습니다! Z 오프셋(nZoffset_mm)을 조정하세요.');
    disp('Tip: sources.mat의 src_z_mm 범위와 PCI Z 범위를 비교하여 오프셋 결정');
end

%% ================ Total PCI Intensity 플롯 ================
disp('Plotting Total PCI Intensity with Vessel Overlay...');

figure('Position', [30 331 780 480], 'color', 'w');
hAx1 = subplot(1,1,1);
hAx2 = axes('Position', hAx1.Position);
linkaxes([hAx1, hAx2]);

% 혈관 이미지 (배경)
hAx1_imagesc = imagesc(hAx1, stG_vessel_itp.aX*1e3, stG_vessel_itp.aZ*1e3, mVessel_itp);
colormap(hAx1, 'gray');
caxis(hAx1, [0, prctile(mVessel_itp(:), 99)]);  % 99 퍼센타일로 클리핑

% PCI 이미지 (오버레이)
bLog = 1;
if bLog
    mPCI_final = log10(mPCI_itp_total + 1e-10);
    nDR = [prctile(mPCI_final(:), 50), prctile(mPCI_final(:), 99.5)];  % 자동 동적 범위
else
    mPCI_final = max(mPCI_itp_total, 0);
    nDR = [0, prctile(mPCI_final(:), 99)];
end

hAx2_imagesc = imagesc(hAx2, stG_pci_itp.aX*1e3, stG_pci_itp.aZ*1e3, mPCI_final);
hAx2_imagesc.AlphaData = (max(hAx2_imagesc.CData, nDR(1)) - nDR(1)) / (nDR(2) - nDR(1));
colormap(hAx2, 'parula');
caxis(hAx2, nDR);

% 축 설정
hAx1.Color = 'k';
hAx2.Visible = 'off';
axis(hAx1, 'equal'); axis(hAx1, 'tight');
axis(hAx2, 'equal'); axis(hAx2, 'tight');
xlabel(hAx1, 'x (mm)'); ylabel(hAx1, 'z (mm)');

hAx1_cb = colorbar(hAx1);
hAx2_cb = colorbar(hAx2);
hAx1_cb.Visible = 'off';
hAx2_cb.Label.String = 'Total PCI intensity';
hTitle = title(hAx1, [sFolderName ' - Vessel Slice #' num2str(SLICE_IDX)]);
hAx1.FontSize = 15;
hAx2.FontSize = 15;
axis(hAx1, aFOV); axis(hAx2, aFOV);

if bLog
    hAx2_cb.Ticks = floor(nDR(1)):1:ceil(nDR(2));
    for tkidx = 1:numel(hAx2_cb.Ticks)
        hAx2_cb.TickLabels{tkidx} = ['10^{' num2str(hAx2_cb.Ticks(tkidx)) '}'];
    end
end

%% ================ 비디오 생성 ================
bVideo = 1;

if bVideo
    % 비디오 출력 폴더
    sVideoDir = fullfile(dataFolder, 'PciData');
    if ~isfolder(sVideoDir); mkdir(sVideoDir); end

    sVideoName = fullfile(sVideoDir, ['PCI_VesselOverlay_slice' num2str(SLICE_IDX,'%04d') '_eig' num2str(nPci_Eig_s) 'to' num2str(nPci_Eig_e)]);
    disp(['Creating... ' sVideoName '.mp4']);
    
    vidObj = VideoWriter(sVideoName, 'MPEG-4');
    vidObj.FrameRate = 5;  % 시뮬레이션 데이터용 (버스트 수가 적음)
    open(vidObj);
    
    % Figure Object
    figure('Position', [800 331 780 480], 'color', 'w');
    hAx1 = subplot(1,1,1);
    hAx2 = axes('Position', hAx1.Position);
    linkaxes([hAx1, hAx2]);
    
    % 혈관 이미지 (배경)
    hAx1_imagesc = imagesc(hAx1, stG_vessel_itp.aX*1e3, stG_vessel_itp.aZ*1e3, mVessel_itp);
    colormap(hAx1, 'gray');
    caxis(hAx1, [0, prctile(mVessel_itp(:), 99)]);
    
    % PCI 이미지 (오버레이) 초기화
    hAx2_imagesc = imagesc(hAx2, stG_pci_itp.aX*1e3, stG_pci_itp.aZ*1e3, zeros(stG_pci_itp.nZdim, stG_pci_itp.nXdim));
    colormap(hAx2, 'parula');
    
    % 축 설정
    hAx1.Color = 'k';
    hAx2.Visible = 'off';
    axis(hAx1, 'equal'); axis(hAx1, 'tight');
    axis(hAx2, 'equal'); axis(hAx2, 'tight');
    xlabel(hAx1, 'x (mm)'); ylabel(hAx1, 'z (mm)');
    hAx1_cb = colorbar(hAx1);
    hAx2_cb = colorbar(hAx2);
    hAx1_cb.Visible = 'off';
    hAx2_cb.Label.String = 'PCI intensity';
    hTitle = title(hAx1, '');
    hAx1.FontSize = 15;
    hAx2.FontSize = 15;
    
    % PCI 동적 범위 설정
    if bLog
        vPCI_final = log10(vPCI_itp_zxb + 1e-10);
        nDR_video = [prctile(vPCI_final(:), 70), prctile(vPCI_final(:), 99.5)];
    else
        vPCI_final = max(vPCI_itp_zxb, 0);
        nDR_video = [0, prctile(vPCI_final(:), 99)];
    end
    caxis(hAx2, nDR_video);
    
    % 프레임 루프
    nNumBursts = size(vPCI_final, 3);
    for bidx = 1:nNumBursts
        % PCI 프레임 업데이트
        hAx2_imagesc.CData = vPCI_final(:,:,bidx);
        hAx2_imagesc.AlphaData = (max(hAx2_imagesc.CData, nDR_video(1)) - nDR_video(1)) / (nDR_video(2) - nDR_video(1));
        
        % 타이틀 업데이트
        hTitle.String = [sFolderName ' - Burst ' num2str(bidx) '/' num2str(nNumBursts)];
        
        % 동적 범위 틱 레이블
        if bLog
            hAx2_cb.Ticks = floor(nDR_video(1)):0.5:ceil(nDR_video(2));
            for tkidx = 1:numel(hAx2_cb.Ticks)
                hAx2_cb.TickLabels{tkidx} = ['10^{' num2str(hAx2_cb.Ticks(tkidx)) '}'];
            end
        end
        
        axis(hAx1, aFOV); axis(hAx2, aFOV);
        drawnow;
        
        currFrame = getframe(gcf);
        writeVideo(vidObj, currFrame);
    end
    
    close(vidObj);
    disp(['Video saved: ' sVideoName '.mp4']);
end

disp('Done!');


%% ================ Helper Function: readNPY ================
function data = readNPY(filename)
% readNPY  Read NumPy .npy file into MATLAB array
%   data = readNPY(filename) reads the numpy array from the file.
%   Supports common dtypes: float32, float64, int16, int32, int64, uint8

    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end
    
    % Read magic string
    magic = fread(fid, 6, 'uint8');
    if ~isequal(magic', [147, 78, 85, 77, 80, 89])  % "\x93NUMPY"
        fclose(fid);
        error('Not a valid .npy file');
    end
    
    % Read version
    version = fread(fid, 2, 'uint8');
    
    % Read header length
    if version(1) == 1
        headerLen = fread(fid, 1, 'uint16');
    else
        headerLen = fread(fid, 1, 'uint32');
    end
    
    % Read header (Python dict as string)
    header = char(fread(fid, headerLen, 'char')');
    
    % Parse dtype
    dtypeMatch = regexp(header, '''descr'':\s*''([^'']+)''', 'tokens');
    if isempty(dtypeMatch)
        fclose(fid);
        error('Cannot parse dtype from header');
    end
    dtypeStr = dtypeMatch{1}{1};
    
    % Parse shape
    shapeMatch = regexp(header, '''shape'':\s*\(([^\)]+)\)', 'tokens');
    if isempty(shapeMatch)
        fclose(fid);
        error('Cannot parse shape from header');
    end
    shapeStr = shapeMatch{1}{1};
    shape = str2num(['[' strrep(shapeStr, ',', ' ') ']']); %#ok<ST2NM>
    if isempty(shape)
        shape = 1;
    end
    
    % Parse fortran order
    fortranMatch = regexp(header, '''fortran_order'':\s*(True|False)', 'tokens');
    if ~isempty(fortranMatch)
        fortranOrder = strcmp(fortranMatch{1}{1}, 'True');
    else
        fortranOrder = false;
    end
    
    % Map dtype to MATLAB
    % Remove byte order character if present
    if dtypeStr(1) == '<' || dtypeStr(1) == '>' || dtypeStr(1) == '|'
        dtypeStr = dtypeStr(2:end);
    end
    
    switch dtypeStr
        case 'f4'
            matlabType = 'single';
        case 'f8'
            matlabType = 'double';
        case 'i2'
            matlabType = 'int16';
        case 'i4'
            matlabType = 'int32';
        case 'i8'
            matlabType = 'int64';
        case 'u1'
            matlabType = 'uint8';
        case 'u2'
            matlabType = 'uint16';
        case 'u4'
            matlabType = 'uint32';
        case 'b1'
            matlabType = 'uint8';  % boolean
        otherwise
            fclose(fid);
            error('Unsupported dtype: %s', dtypeStr);
    end
    
    % Read data
    numElements = prod(shape);
    data = fread(fid, numElements, matlabType);
    fclose(fid);
    
    % Reshape
    if numel(shape) > 1
        if fortranOrder
            data = reshape(data, shape);
        else
            % C order (row-major) -> need to transpose
            data = reshape(data, fliplr(shape));
            data = permute(data, ndims(data):-1:1);
        end
    end
end
