% Sua Bae
%    3/26/2021
%
function binsave(Data, sPath, sPrecision)
    fid = fopen(sPath,'wb');
    fwrite(fid,Data,sPrecision);
    fclose(fid);
end