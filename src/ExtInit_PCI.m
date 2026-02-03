% Written by Sua Bae
%   3/1/2022
%       P.CAV.aRfSigPow => P.CAV.aSigPow
%   2/22/2022
%    initialize CUDA PCI Beamformer
%       gen kernel, alloc mem
%       matched with InitCuda_PCI.m and ExtRun_PCI.m
%
% ---- PATCH (2026-02-03) ----
% Purpose: make ExtInit_PCI robust for simulation-only pipelines where
% runtime plotting fields (e.g., P.CAV.aSigPow) may not exist yet.
% - Initializes missing runtime fields safely
% - Makes plotting optional (won't crash headless / no-stG cases)
% ----------------------------
function ExtInit_PCI(varargin)
    disp('# ExtInit_PCI.m: running...');
    global P

    % -----------------------------
    % 0) Basic guards / defaults
    % -----------------------------
    if isempty(P) || ~isstruct(P)
        error('Global P is empty or not a struct. Load P.mat into global P before calling ExtInit_PCI.');
    end
    if ~isfield(P,'CAV') || ~isstruct(P.CAV)
        P.CAV = struct();
    end
    if ~isfield(P,'FUS') || ~isstruct(P.FUS)
        P.FUS = struct();
    end

    % Optional plotting flag: default = true
    if ~isfield(P.CAV,'bPlot') || isempty(P.CAV.bPlot)
        P.CAV.bPlot = true;
    end
    bPlot = logical(P.CAV.bPlot);
    if nargin >= 1 && ~isempty(varargin{1})
        bPlot = logical(varargin{1});
        P.CAV.bPlot = bPlot;
    end

    % Session name fallback
    if ~isfield(P,'sSessionName') || isempty(P.sSessionName)
        P.sSessionName = 'PCI_session';
    end

    % Burst counter (runtime)
    if ~isfield(P.CAV,'nBurstCount') || isempty(P.CAV.nBurstCount)
        P.CAV.nBurstCount = 0;
    end

    % -----------------------------
    % 1) Reset counter if needed
    % -----------------------------
    if P.CAV.nBurstCount ~= 0
        disp(['nBurstCount was ' num2str(P.CAV.nBurstCount)]);
        P.CAV.nBurstCount = 0; % start new session!
        disp(['Starting a new session from nBurstCount=' num2str(P.CAV.nBurstCount)]);
    end

    % -----------------------------
    % 2) Initialize signal power history (runtime)
    % -----------------------------
    nNumBurst = 1;
    if isfield(P.FUS,'nNumBurst') && ~isempty(P.FUS.nNumBurst) && isnumeric(P.FUS.nNumBurst)
        nNumBurst = double(P.FUS.nNumBurst);
        if nNumBurst < 1, nNumBurst = 1; end
    end

    if ~isfield(P.CAV,'aSigPow') || isempty(P.CAV.aSigPow) || ~isnumeric(P.CAV.aSigPow)
        P.CAV.aSigPow = zeros(1, nNumBurst, 'single');
    else
        % Ensure length matches nNumBurst (pad or trim)
        if numel(P.CAV.aSigPow) < nNumBurst
            tmp = zeros(1, nNumBurst, 'single');
            tmp(1:numel(P.CAV.aSigPow)) = single(P.CAV.aSigPow(:));
            P.CAV.aSigPow = tmp;
        elseif numel(P.CAV.aSigPow) > nNumBurst
            P.CAV.aSigPow = single(P.CAV.aSigPow(1:nNumBurst));
        else
            P.CAV.aSigPow = single(P.CAV.aSigPow(:).'); % row
        end
    end

    % -----------------------------
    % 3) Plot setup (optional)
    % -----------------------------
    % If you are running purely to generate PCI outputs without GUI,
    % call ExtInit_PCI(false) or set P.CAV.bPlot = false.
    if ~bPlot
        % Ensure handles exist (so ExtRun_PCI can "isgraphics" check safely)
        P.CAV.hFig = [];
        P.CAV.hAx1 = [];
        P.CAV.hAx1_imagesc = [];
        P.CAV.hAx1_title = [];
        P.CAV.hAx2 = [];
        P.CAV.hAx2_plot = [];
        return;
    end

    % Only plot if grid exists
    hasGrid = isfield(P.CAV,'stG') && isstruct(P.CAV.stG) ...
        && isfield(P.CAV.stG,'aX') && isfield(P.CAV.stG,'aZ') ...
        && isfield(P.CAV.stG,'nZdim') && isfield(P.CAV.stG,'nXdim');

    try
        P.CAV.hFig = figure('Name',P.sSessionName,...
            'NumberTitle','off','Visible','on',...
            'Position',[100, 100, 950, 500]);  % [left bottom width height]
    catch
        % Headless or figure not available -> disable plotting safely
        warning('ExtInit_PCI:FigureFailed','Figure creation failed; continuing without plotting.');
        P.CAV.bPlot = false;
        P.CAV.hFig = [];
        P.CAV.hAx1 = []; P.CAV.hAx1_imagesc = []; P.CAV.hAx1_title = [];
        P.CAV.hAx2 = []; P.CAV.hAx2_plot = [];
        return;
    end

    % - image axis (left)
    P.CAV.hAx1 = subplot(1,2,1);

    if hasGrid
        P.CAV.hAx1_imagesc = imagesc(P.CAV.hAx1, ...
            P.CAV.stG.aX*1e3, P.CAV.stG.aZ*1e3, zeros(P.CAV.stG.nZdim,P.CAV.stG.nXdim,'single'));
        xlabel(P.CAV.hAx1, 'x (mm)'); ylabel(P.CAV.hAx1, 'z (mm)');
        axis(P.CAV.hAx1, 'equal'); axis(P.CAV.hAx1, 'tight'); P.CAV.hAx1.Color = 'k';
        P.CAV.hAx1_title = title(P.CAV.hAx1, ['Power cavitation map, bidx=' num2str(P.CAV.nBurstCount)]);
    else
        % If no grid, show placeholder to avoid crash
        P.CAV.hAx1_imagesc = imagesc(P.CAV.hAx1, zeros(10,10,'single'));
        axis(P.CAV.hAx1, 'image'); axis(P.CAV.hAx1, 'tight'); P.CAV.hAx1.Color = 'k';
        P.CAV.hAx1_title = title(P.CAV.hAx1, 'Power cavitation map (grid missing)');
        xlabel(P.CAV.hAx1, ''); ylabel(P.CAV.hAx1, '');
    end

    % - signal power (right)
    P.CAV.hAx2 = subplot(1,2,2);
    P.CAV.hAx2_plot = plot(P.CAV.hAx2, 1:numel(P.CAV.aSigPow), P.CAV.aSigPow);
    xlim(P.CAV.hAx2, [1, nNumBurst]);
    xlabel(P.CAV.hAx2, 'burst idx'); ylabel(P.CAV.hAx2, 'RF signal power');
    grid(P.CAV.hAx2, 'on'); grid(P.CAV.hAx2, 'minor');
end
