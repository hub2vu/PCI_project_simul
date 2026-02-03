% Written by Sua Bae
%   3/1/2022
%       update for offline proc (P.bProcess == 2)
%       P.CAV.aRfSigPow => P.CAV.aSigPow (calc of SVD filtered)
%   2/22/2022
%    initialize CUDA PCI Beamformer
%       gen kernel, alloc mem
%       matched with InitCuda_PCI.m and ExtInit_PCI.m
%   
function ExtRun_PCI(mRcvData_org)


    disp('# ExtRun_PCI.m: running...');
    global P g kBF_PCI mBfData_zxp mTxDelay_zx_m
       
    
    %%% 6. Update burst index 
    P.CAV.nBurstCount = P.CAV.nBurstCount + 1;
    
    % import parameters    
    stRfInfo    = P.CAV.stRfInfo;
    stG         = P.CAV.stG;
    
    
    %%% 1. Cut data
    mRfData_spc = mRcvData_org(1:stRfInfo.nSample*P.FUS.nNumPulse, :); % size: sample x pulse (usually, 100) x channel
    
    %%% 2. Processing Data ------------------------------------
    if P.bProcess  % if it's either 1 or 2 (online or offline)      
        disp(['Beamforming...' num2str(stRfInfo.nSample) 'samples '  num2str(stRfInfo.nChannel) 'channels ' num2str(stG.nPdim) 'frames']);
        tic;
        
        aRfData_temp_gpu = gpuArray(single(mRfData_spc));% load mRfData_spc onto GPU
        wait(g);

        % 1) reshape data for faster access
        vRfData_scp_gpu = permute(reshape(aRfData_temp_gpu,[stRfInfo.nSample, P.FUS.nNumPulse, stRfInfo.nChannel]),[1,3,2]); % (s*p,c) -> (s,c,p) % (sample, channel, pulse)
        wait(g); 
        
        clear aRfData_temp_gpu;

        % 2) beamforming with CUDA kernel
        [mBfData_zxp] = feval(kBF_PCI, mBfData_zxp, vRfData_scp_gpu, mTxDelay_zx_m); % call CUDA kernel
        wait(g);  
        
        % 3) SVD filtering
        % dimension of mBfData_zxp : [stG.nZdim*stG.nXdim, stG.nPdim]      -> casorati   
        disp(['SVD filtering...eig=' num2str(P.CAV.nEig_s) ':' num2str(P.CAV.nEig_e)]);
        [~,~,mV] = svd(ctranspose(mBfData_zxp)*mBfData_zxp,0);
        mV_f = mV(:, P.CAV.nEig_s:P.CAV.nEig_e);
        vFiltered_zxp = reshape(mBfData_zxp*mV_f*ctranspose(mV_f), [stG.nZdim,stG.nXdim,stG.nPdim]);     
        wait(g); 
        
        % 4) take absolute and accumulate across pulses (frames) + gather
        mPCI_zx = gather(sum(abs(vFiltered_zxp).^2,3));
        wait(g); 
        
        % 5) calc signal power
        P.CAV.aSigPow(P.CAV.nBurstCount) = mean(mPCI_zx(:));
            
        nTime = toc;
        disp(['   processing time: ' num2str(nTime) ' sec']);
    end
    
    
    %%% 3. Display Data ------------------------------------
    if P.bProcess && P.bDisplay
        
        % - image
        mPCI_zx_db = 10*log10(abs(mPCI_zx/max(mPCI_zx(:))));
        P.CAV.hAx1_imagesc.CData = mPCI_zx_db;
        P.CAV.hAx1_title.String = ['Power cavitation map, bidx=' num2str(P.CAV.nBurstCount)];
        
        % - signal power        
        set(P.CAV.hAx2_plot,'XData',1:numel(P.CAV.aSigPow),'YData',P.CAV.aSigPow);
        
        drawnow;
    end
        
    
    %%% 4. Save Data ---------------------------------------------------
    if (P.bProcess~=2) % when it's not offline process
        if (P.bSaveData == 1)
            tic;
        
            % Save RF Data        
            %  make folder 
            sFolder = [P.sSaveFolder '/' P.sSessionName '/' 'RfData/'];
            if ~exist(sFolder,'dir'); mkdir(sFolder); end  

            %  save RF data as .bin (faster than .mat)
            sFilename_rf = ['RfData_spc' '_bidx' num2str(P.CAV.nBurstCount), ...
                                         '_' datestr(now,'yymmdd_HHMM_SS_FFF') '.bin']; 
            disp([sFolder '/' sFilename_rf]);        
            disp(['Saving Data...:' sFilename_rf]);
            fid = fopen([sFolder sFilename_rf], 'wb');
            fwrite(fid, mRfData_spc, 'int16'); % size: sample x pulse x channel
            fclose(fid);
            
            
            % Save Processed Data
            if P.bProcess == 1
                stPCI.stG = stG;
                stPCI.mPCI_zx = mPCI_zx;
                sFilename_pci = ['stPCI' '_bidx' num2str(P.CAV.nBurstCount), ...
                                         '_' datestr(now,'yymmdd_HHMM_SS_FFF') '.mat']; 
                save([sFolder '/' sFilename_pci],'stPCI'); 
            end
        
            nTime = toc;
            disp(['   data saving time: ' num2str(nTime) ' sec']);
        end

    else
        % If it's offline process,
        vPCI_zxb = evalin('base','vPCI_zxb');
        vPCI_zxb(:,:,P.CAV.nBurstCount) = mPCI_zx;
        assignin('base','vPCI_zxb',vPCI_zxb);
    end


        
    
    disp('# ExtRun_PCI.m: done...');
end
