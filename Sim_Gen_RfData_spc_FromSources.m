% Sim_Gen_RfData_spc_FromSources.m
clear; close all;
addpath('src');

SESSION_DIR = '../Data_tus/01_sim';   % 우리가 만든 synthetic 세션 폴더
load(fullfile(SESSION_DIR,'sources.mat'));  % Python이 만든 소스

S = load(fullfile(SESSION_DIR,'P.mat'));    % 너가 복사해둔 P.mat
P = S.P;

stRfInfo = P.CAV.stRfInfo;
stTrans  = P.stTrans;

outDir = fullfile(SESSION_DIR,'RfData');
if ~exist(outDir,'dir'); mkdir(outDir); end

% ======== 여기부터 RF 생성 (간단/안정 버전) ========
c = 1540;

nCh = stRfInfo.nChannel;
nSample = stRfInfo.nSample;
nPulse = P.FUS.nNumPulse;

% 샘플링 주파수는 P에 있을 수도, 없을 수도 있음.
% 없으면 너가 Top02_01에서 쓰는 값과 맞춰야 함.
fs = 40e6;
if isfield(stRfInfo,'nFs') && ~isempty(stRfInfo.nFs), fs = stRfInfo.nFs; end
t = (0:nSample-1).'/fs;


% 128ch element 좌표 (P.stTrans에 맞춰 꺼내기)
if isfield(stTrans,'aElePosX_m')
    ele_x = stTrans.aElePosX_m(:);
    ele_z = stTrans.aElePosZ_m(:);

elseif isfield(stTrans,'mElementPos')
    ele_x = stTrans.mElementPos(:,1);
    ele_z = stTrans.mElementPos(:,3);

elseif isfield(stTrans,'aElePos')
    a = stTrans.aElePos;
    if isvector(a)  % (128x1) 같은 케이스: x만 있음
        ele_x = a(:);
        ele_z = zeros(size(ele_x));
    else
        % 혹시 (N x 3) 이상으로 들어있으면 x,z를 꺼내기
        ele_x = a(:,1);
        if size(a,2) >= 3
            ele_z = a(:,3);
        else
            ele_z = zeros(size(ele_x));
        end
    end

else
    error('P.stTrans에서 element 좌표 필드명을 찾지 못함. stTrans 구조 확인 필요');
end

% (선택) 렌즈 보정이 있으면 z 기준면을 살짝 당겨서 time-of-flight에 반영
if isfield(stTrans,'nLensCorr_m') && ~isempty(stTrans.nLensCorr_m)
    ele_z = ele_z - double(stTrans.nLensCorr_m);
end

% sanity check
if numel(ele_x) ~= nCh
    error('element 개수(%d)와 nChannel(%d)이 다릅니다.', numel(ele_x), nCh);
end


src_x = double(src_x_mm(:))*1e-3;
src_z = double(src_z_mm(:))*1e-3;
src_amp = double(src_amp(:));
src_type = double(src_type(:)); % 1=inertial,0=stable
nSrc = numel(src_x);

f0 = 1.5e6;

nBurst = 30;
for bidx = 1:nBurst
    rf_all = zeros(nSample*nPulse, nCh, 'single');

    % burst마다 소스 진폭 약간 랜덤 (현실성)
    jitter_b = 0.8 + 0.4*rand(nSrc,1);
    phi_b = 2*pi*rand(nSrc,1);

    % 거리/지연 (vectorized)
    dx = src_x - ele_x.';
    dz = src_z - ele_z.';
    dist = sqrt(dx.^2 + dz.^2) + 1e-9;
    tau  = dist / c;
    att  = 1 ./ dist;

    for pidx = 1:nPulse
        rf = zeros(nSample, nCh, 'single');

        % pulse마다 약간 변동
        jitter = jitter_b .* (0.9 + 0.2*rand(nSrc,1));

        for ch = 1:nCh
            sig = zeros(nSample,1,'single');

            idxS = (src_type==0);
            if any(idxS)
                tauS = tau(idxS,ch);
                aS   = src_amp(idxS).*jitter(idxS).*att(idxS,ch);
                for k=1:numel(tauS)
                    tt = t - tauS(k);
                    w = single((tt>=0) & (tt<=15e-6));
                    s = sin(2*pi*(f0/2)*tt + phi_b(k)) + 0.6*sin(2*pi*f0*tt + phi_b(k));
                    sig = sig + single(aS(k)) .* single(s) .* w;
                end
            end

            idxI = (src_type==1);
            if any(idxI)
                tauI = tau(idxI,ch);
                aI   = src_amp(idxI).*jitter(idxI).*att(idxI,ch);
                for k=1:numel(tauI)
                    tt = t - tauI(k);
                    w = single((tt>=0) & (tt<=10e-6));
                    sig = sig + single(aI(k)) .* randn(size(tt),'single') .* w;
                end
            end

            rf(:,ch) = sig;
        end

        i0 = (pidx-1)*nSample + 1;
        rf_all(i0:i0+nSample-1,:) = rf_all(i0:i0+nSample-1,:) + rf;
    end

    mx = max(abs(rf_all(:))) + 1e-9;
    scale = 0.8 * 32767 / mx;
    rf_i16 = int16(rf_all * scale);

    fname = sprintf('RfData_spc_%03d.bin', bidx);
    fid = fopen(fullfile(outDir,fname),'w');
    fwrite(fid, rf_i16, 'int16');
    fclose(fid);

    fprintf('saved %s (scale=%.3g)\n', fname, scale);
end
