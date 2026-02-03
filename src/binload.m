% Written by Sua Bae
%    3/16/2022
%       - updated for the option without aSize
%    3/26/2021
%
function Data_reshaped = binload(sPath, sPrecision, aSize)
    
    fid = fopen(sPath,'rb');
    Data = fread(fid,sPrecision);
    fclose(fid);
    
    if nargin < 3
        Data_reshaped = Data;
    else
        Data_reshaped = reshape(Data,aSize);
    end
    
end