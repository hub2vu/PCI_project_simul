% Written by Sua Bae (07/15/2022)
%   Filtering for removing grid artifacts (periodic noise)
% Updated by Sua Bae (07/17/2022)
%   Add mMask2 (big gaussian weight) to preserve high freq
function mImg_filt = rmGridArtifact(mImg, nIntp, n2ndInterpFactor, bPlot)

    if ~exist('bPlot','var')||isempty(bPlot)
        bPlot = 0;
    end

    if bPlot
        figure; 
        subplot(2,3,1); imagesc(mImg); caxis([0 1]);
    end
    mFft = fftshift(fft2(mImg));
    
    
    aKx = linspace(-0.5,0.5,size(mFft,2));
    aKy = linspace(-0.5,0.5,size(mFft,1));
    [mY,mX] = ndgrid(aKy,aKx);
    
    if bPlot
      subplot(2,3,2); imagesc(aKx,aKy,20*log10(abs(mFft))); colorbar; caxis([-100 0]+max(20*log10(abs(mFft(:)))));
    end
    mMask1 = zeros(size(mFft));
    
    aPx = linspace(-0.5,0.5,nIntp*n2ndInterpFactor+1); 
    aPy = linspace(-0.5,0.5,nIntp*n2ndInterpFactor+1); 
    [mPy, mPx] = ndgrid(aPy,aPx);
    mNotch = [mPx(:),mPy(:)];
    [~, index]=ismember(mNotch,[0 0],'rows');
    mNotch([find(index)],:) = [];
    aStd = [1/(nIntp*n2ndInterpFactor)/4, 1/(nIntp*n2ndInterpFactor)/4]; %*1.5;

    for nidx = 1:size(mNotch,1)
        mMask1 = mMask1 + exp(-(mX-mNotch(nidx,1)).^2/(2*aStd(1)^2) - (mY-mNotch(nidx,2)).^2/(2*aStd(2)^2));
        % mMask((mX-mNotch(nidx,1)).^2/(aStd(1)^2) + (mY-mNotch(nidx,2)).^2/(aStd(2)^2) < 1) = 1;
    end
    mMask2 = exp(-mX.^2/0.5 - mY.^2/0.5);
    mMask = mMask2.*mMask1;
    mMask_norm = (mMask - min(mMask(:)))/(max(mMask(:))-min(mMask(:)));
    mMask_inv = 1 - mMask_norm;
    if bPlot
        subplot(2,3,3); imagesc(mMask_inv); colorbar;
    end
    
    mFft = mFft.*mMask_inv;
    
    if bPlot
      subplot(2,3,5); imagesc(aKx,aKy,20*log10(abs(mFft))); colorbar; caxis([-100 0]+max(20*log10(abs(mFft(:)))));
    end
    
    mImg_filt = abs(ifft2(fftshift(mFft)));
    
    if bPlot
      subplot(2,3,4); imagesc(mImg_filt);caxis([0 1]);
    end
    
end