% Written by Sua Bae (08/24/2022)
%   This function is for interpolation of the image for isotropic pixels
%
%   mImg:           input image
%   stG:            grid information of the input image
%   nPixelSize_m:   (meter) size of isotropic pixel 
%
%   mImg_itp:       output image
%   stG_itp:        interpolated grid
function [mImg_itp, stG_itp] = interpImg(mImg, stG, nPixelSize_m)

    % original image grid points
    [mZ,mX] = ndgrid(stG.aZ,stG.aX);

    % query image grid points
    aXq = stG.aX(1):nPixelSize_m:stG.aX(end);
    aZq = stG.aZ(1):nPixelSize_m:stG.aZ(end);
    [mZq,mXq] = ndgrid(aZq,aXq);

    % interpolation
%     mImg_itp = interpn(mZ,mX,mImg,mZq,mXq,'spline');
    mImg_itp = interpn(mZ,mX,mImg,mZq,mXq); % 1/29/2024

    stG_itp.aX = aXq;
    stG_itp.aZ = aZq;
    stG_itp.dx = nPixelSize_m;
    stG_itp.dz = nPixelSize_m;
    stG_itp.nXdim = numel(aXq);
    stG_itp.nZdim = numel(aZq);

end