% Written by Sua Bae
%   3/1/2022
%       update for offline proc (P.bProcess == 2)
%       P.CAV.aRfSigPow => P.CAV.aSigPow
%   2/22/2022
%    initialize CUDA PCI Beamformer
%       gen kernel, alloc mem
%       matched with InitCuda_PCI.m and ExtInit_PCI.m
   
function InitCuda_PCI

    global P kBF_PCI mBfData_zxp mTxDelay_zx_m
    
    disp('InitCuda_PCI.m running...');
    
    
    stRfInfo = P.CAV.stRfInfo;
    stTrans = P.stTrans;
        
    nRxFnum = 1;
    
    % BF image Grid info
    nPitch = stTrans.aElePos(2)-stTrans.aElePos(1);
    % - lateral pixels 
    stG.dx = nPitch * P.CAV.aImgRes_xz_pitch(1);
    stG.aX = stTrans.aElePos(1):stG.dx:stTrans.aElePos(end);
    stG.nXdim = numel(stG.aX);
    % - axial pixels
    stG.dz = nPitch * P.CAV.aImgRes_xz_pitch(2);
    stG.aZ = P.CAV.startDepth*P.stTrans.nWaveLength:stG.dz:P.CAV.endDepth*P.stTrans.nWaveLength; % axial axis [m] centered at focus
    stG.nZdim = numel(stG.aZ);
    % - number of pulses
    stG.nPdim = P.FUS.nNumPulse; 
            
    % allocate GPU Array for output
    mBfData_zxp = single(zeros(stG.nZdim*stG.nXdim, stG.nPdim,'gpuArray')); % size: z*x, p for casorati matrix

    % compute TXPD
    % - PData structure array.
    PData(1).PDelta = [stG.dx, 0, stG.dz]/stTrans.nWaveLength; % [wl] xyz % [1, 0, 1];% xyz
    PData(1).Size(1) = stG.nZdim; % ceil((P.endDepth-P.startDepth)/PData(1).PDelta(3));
    PData(1).Size(2) = stG.nXdim; % ceil((P.imgWidth)/PData(1).PDelta(1));
    PData(1).Size(3) = 1;
    PData(1).Origin = [stG.aX(1),0,stG.aZ(1)]/stTrans.nWaveLength; % [wl] % [-(PData(1).Size(2)/2)*PData(1).PDelta(1),0,P.startDepth];
    if (P.bProcess ~= 2) % if it's not offline processing (cause there will be no TX in offline proc)
        TX = evalin('base', 'TX');
        vTXPD_zxo = double(computeTXPD(TX(P.IMG.numRays+1),PData)); % TXPD of FUS beam
        P.FUS.mBeampattern = vTXPD_zxo(:,:,1);
      % figure;imagesc(stG.aX*1e3,stG.aZ*1e3,vTXPD_zxo(:,:,1)); title('beam pattern'); xlabel('x(mm)'); ylabel('z(mm)'); axis equal tight;
      % figure;imagesc(stG.aX*1e3,stG.aZ*1e3,vTXPD_zxo(:,:,2)/16*stTrans.nWaveLength*1e3); title('TX delay (mm)'); xlabel('x(mm)'); ylabel('z(mm)'); axis equal tight;
    
        % load Tx delay in GPU mem    
        mTxDelay_zx_m = gpuArray(single(squeeze(vTXPD_zxo(:,:,2))/16*stTrans.nWaveLength));
        P.CAV.mTxDelay_zx_m = gather(mTxDelay_zx_m);
          % figure;imagesc(stG.aX*1e3,stG.aZ*1e3,mTxDelay_zx*1e3); title('TX delay (mm)'); xlabel('x(mm)'); ylabel('z(mm)'); axis equal tight;
    else
        % if processing offline, mTxDelay will be in P
        mTxDelay_zx_m = gpuArray(P.CAV.mTxDelay_zx_m); % load onto GPU
    end
    
    
    % compile cuda code
    sCudaCodeName = 'Beamformer_PCI.cu';
    if P.CAV.bCompile
        disp(['compiling cuda code..(' sCudaCodeName ')']);
        cwd = pwd; cd([cwd '/' P.CAV.sCodePath]);
        system(['nvcc -ptx ' sCudaCodeName  ' -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC"' ]);
        cd(cwd);
    else
        disp(['using already complied code..(' sCudaCodeName(1:end-3) '.ptx)']);
    end
    
    % setup cuda kernel
    disp(['setting kernel..(' sCudaCodeName ')']);
    kBF_PCI = parallel.gpu.CUDAKernel([sCudaCodeName(1:end-3) '.ptx'],[sCudaCodeName(1:end-3) '.cu'],'_Beamformer_PCI');
    kBF_PCI.ThreadBlockSize =[512,1,1];%[16,16,1];
    kBF_PCI.GridSize = [ceil(stG.nXdim/kBF_PCI.ThreadBlockSize(1)),stG.nZdim,stG.nPdim];
    setConstantMemory(kBF_PCI,...
        'trans_aElePos',        single(stTrans.aElePos),...   % [m] transducer element position in x
        'trans_nNumEle',        int32(stTrans.nNumEle),...    % num of txdcr elements
        'rf_nSdim',             int32(stRfInfo.nSample),...       % num of samples of RF data
        'rf_nCdim',             int32(stRfInfo.nChannel),...      % num of channels of RF data
        'rf_nPdim',             int32(P.FUS.nNumPulse),...        % num of pulses (num of tx/rx) of RF data = num of focused pulses for each burst
        'bf_nXdim',              int32(stG.nXdim),...              % num of pixels in x of BF image
        'bf_nZdim',              int32(stG.nZdim),...              % num of pixels in z of BF image
        'bf_dx',                 single(stG.dx),...                % [m] x pixel size 
        'bf_dz',                 single(stG.dz),...                % [m] z pixel size 
        'bf_nXstart',            single(stG.aX(1)),...             % [m] starting x coordinate
        'bf_nZstart',            single(stG.aZ(1)), ...            % [m] starting z coordinate
        'nFs',                  single(stRfInfo.nFs),...          % [Hz]  sampling frequency of RF data
        'nSoundSpeed',          single(stRfInfo.nSoundSpeed),...  % [m/s] sound speed of material
        'nRxFnum',              single(nRxFnum),...               % f-number for receive beamforming
        'nLensCorr',            single(stTrans.nLensCorr_m),...   % [m] lens correction valuem, will be ADDED to the round-trip delay
        'nOffset_m',            single(P.CAV.startDepth*P.stTrans.nWaveLength));  % [m] offset of acquired Rf Data, will be SUBTRACTED from the round-trip delay (for example, P.startDepth*nWavelength_m), 
    
    stG.nPdim = P.FUS.nNumPulse;

    
    %
    P.CAV.sCudaCodeName = sCudaCodeName;
%     P.CAV.sDir = [];
%     P.CAV.hFig = [];
%     P.CAV.hAx1 = [];
%     P.CAV.hAx2 = [];
    P.CAV.stG = stG;
    P.CAV.nBurstCount = 0; % initialize
    P.CAV.aSigPow = 0; % initialize

    disp('InitCuda_PCI.m done...');
end
