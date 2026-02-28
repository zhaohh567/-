function out = demo_compare_subpixel_methods(folder, numFrames)
% DEMO_COMPARE_SUBPIXEL_METHODS
% Compare subpixel localization methods under DoG detection for drift workflow:
%   Method A: local weighted centroid (7x7, background-subtracted, optional Gaussian weight)
%   Method B: quadratic (parabolic) interpolation on DoG response (3-point in x/y)
%
% Inputs
%   folder    : folder containing single-frame TIFFs (sorted by filename)
%   numFrames : number of frames to process (e.g., 50). default = 50
%
% Output (struct out)
%   out.stats: table-like struct arrays per interval (t -> t+1):
%       .dx_med, .dy_med, .sigx, .sigy, .nMatch for each method
%   out.params: parameters used

if nargin < 2, numFrames = 50; end

% -------------------- Parameters (recommended for your beads scenario) --------------------
p.sigma1   = 1.2;              % DoG small sigma (px)
p.sigma2   = 1.6*p.sigma1;     % DoG large sigma (px)
p.k_thresh = 5.5;              % DoG robust threshold multiplier (5~6)
p.minDist  = 3;                % NMS minimum distance (px)
p.maxMatchDist = 3.0;          % max NN match distance (px), small drift expected

p.winCent  = 7;                % centroid window size (odd)
p.useGaussWeight = true;
p.sigmaW   = 1.2;              % Gaussian weight sigma inside centroid window (px)
% Outlier removal (FIXED threshold for fair comparison)更改
p.resMax = 0.30;   % px, fixed 2D residual threshold (recommend 0.25~0.35)
% p.outlierMAD = 3.0;            % outlier rejection on residuals: |res| > outlierMAD * robustSigma

% -------------------- Read file list --------------------
files = dir(fullfile(folder, '*.tif*'));
assert(~isempty(files), 'No TIFF files found in folder: %s', folder);
[~, idx] = sort({files.name});
files = files(idx);
numFrames = min(numFrames, numel(files));

% Precompute centroid Gaussian weight grid
hC = floor(p.winCent/2);
[xg, yg] = meshgrid(-hC:hC, -hC:hC);
if p.useGaussWeight
    Gw = exp(-(xg.^2 + yg.^2)/(2*p.sigmaW^2));
else
    Gw = ones(size(xg));
end

% Storage: positions per frame, for two methods
posA = cell(numFrames,1); % [x y] weighted centroid
posB = cell(numFrames,1); % [x y] quadratic on DoG

tCentroid = 0; tQuad = 0;

% -------------------- Per-frame detection + subpixel refine --------------------
for t = 1:numFrames
    I = double(imread(fullfile(folder, files(t).name)));

    % DoG
    G1 = gaussBlur2D(I, p.sigma1);
    G2 = gaussBlur2D(I, p.sigma2);
    DoG = G1 - G2;

    % Robust threshold on DoG
    thr = median(DoG(:)) + p.k_thresh * robustSigma(DoG(:));

    % Local maxima (8-neighborhood) on DoG
    m = localMax8(DoG) & (DoG > thr);

    % Exclude borders needed by centroid window and quadratic (needs 1-pixel border for 3x3)
    border = max(hC, 1) + 1;
    m(1:border,:) = false; m(end-border+1:end,:) = false;
    m(:,1:border) = false; m(:,end-border+1:end) = false;

    [yy, xx] = find(m);
    if isempty(xx)
        posA{t} = zeros(0,2);
        posB{t} = zeros(0,2);
        continue;
    end

    % Greedy NMS by DoG peak strength
    vals = DoG(sub2ind(size(DoG), yy, xx));
    [~, ord] = sort(vals, 'descend');
    xx = xx(ord); yy = yy(ord);

    keep = false(size(xx));
    keptXY = zeros(0,2);
    for i = 1:numel(xx)
        if isempty(keptXY)
            keep(i) = true;
            keptXY(end+1,:) = [xx(i), yy(i)];
        else
            d2 = (keptXY(:,1)-xx(i)).^2 + (keptXY(:,2)-yy(i)).^2;
            if all(d2 > p.minDist^2)
                keep(i) = true;
                keptXY(end+1,:) = [xx(i), yy(i)];
            end
        end
    end
    xx = xx(keep); yy = yy(keep);

    % ---- Method A: weighted centroid on raw image ----
    ticA = tic;
    XA = zeros(numel(xx),1);
    YA = zeros(numel(xx),1);
    for i = 1:numel(xx)
        x0 = xx(i); y0 = yy(i);
        patch = I(y0-hC:y0+hC, x0-hC:x0+hC);

        % background estimate (median)
        bg = median(patch(:));
        W = max(patch - bg, 0) .* Gw;

        s = sum(W(:));
        if s <= 0
            XA(i) = x0; YA(i) = y0;
        else
            % absolute coordinate grids
            Xabs = x0 + xg;
            Yabs = y0 + yg;
            XA(i) = sum(sum(W .* Xabs)) / s;
            YA(i) = sum(sum(W .* Yabs)) / s;
        end
    end
    tCentroid = tCentroid + toc(ticA);
    posA{t} = [XA, YA];

    % ---- Method B: quadratic interpolation on DoG response (fast) ----
    ticB = tic;
    XB = zeros(numel(xx),1);
    YB = zeros(numel(xx),1);
    for i = 1:numel(xx)
        x0 = xx(i); y0 = yy(i);

        % 1D parabolic offsets using DoG values
        fm = DoG(y0, x0-1); f0 = DoG(y0, x0); fp = DoG(y0, x0+1);
        dx = parabolicOffset(fm, f0, fp);

        fm = DoG(y0-1, x0); f0 = DoG(y0, x0); fp = DoG(y0+1, x0);
        dy = parabolicOffset(fm, f0, fp);

        XB(i) = x0 + dx;
        YB(i) = y0 + dy;
    end
    tQuad = tQuad + toc(ticB);
    posB{t} = [XB, YB];
end

% -------------------- Per-interval matching + drift + robust scatter --------------------
M = numFrames - 1;

% 更改
A.dx_med = nan(M,1); A.dy_med = nan(M,1);
A.sigx   = nan(M,1); A.sigy   = nan(M,1);
A.nMatch_raw  = zeros(M,1);   % before outlier removal
A.nMatch_keep = zeros(M,1);   % after outlier removal

B = A;
% A.dx_med = nan(M,1); A.dy_med = nan(M,1); A.sigx = nan(M,1); A.sigy = nan(M,1); A.nMatch = zeros(M,1);
% B = A;

for t = 1:M
    % Method A
    [dx, dy] = matchAndDisplacement(posA{t}, posA{t+1}, p.maxMatchDist);
%     [A.dx_med(t), A.dy_med(t), A.sigx(t), A.sigy(t), A.nMatch(t)] = driftAndSigma(dx, dy, p.outlierMAD);
[A.dx_med(t), A.dy_med(t), A.sigx(t), A.sigy(t), A.nMatch_raw(t), A.nMatch_keep(t)] = driftAndSigma(dx, dy, p.resMax);%更改

    % Method B
    [dx, dy] = matchAndDisplacement(posB{t}, posB{t+1}, p.maxMatchDist);
%     [B.dx_med(t), B.dy_med(t), B.sigx(t), B.sigy(t), B.nMatch(t)] = driftAndSigma(dx, dy, p.outlierMAD);
    [B.dx_med(t), B.dy_med(t), B.sigx(t), B.sigy(t), B.nMatch_raw(t), B.nMatch_keep(t)] = driftAndSigma(dx, dy, p.resMax);;%更改
end

% -------------------- Summary + plots --------------------
fprintf('Processed frames: %d (intervals: %d)\n', numFrames, M);
fprintf('Refine time: centroid %.3f s total, quadratic %.3f s total (only refine loops)\n', tCentroid, tQuad);

% Overall comparison metrics (ignore NaNs)
mAx = median(A.sigx(~isnan(A.sigx)));  mAy = median(A.sigy(~isnan(A.sigy)));
mBx = median(B.sigx(~isnan(B.sigx)));  mBy = median(B.sigy(~isnan(B.sigy)));
fprintf('Median robust sigma (x,y): centroid (%.4f, %.4f) px; quadratic (%.4f, %.4f) px\n', mAx, mAy, mBx, mBy);

tt = (1:M)';
figure('Name','Subpixel method comparison');
subplot(2,1,1);
plot(tt, A.sigx, '-o'); hold on; plot(tt, B.sigx, '-o');
ylabel('robust \sigma_x (px)'); grid on; legend('Centroid','Quadratic','Location','best');
title('Per-interval residual scatter (after median drift removal)');

% subplot(2,1,2);
% plot(tt, A.nMatch, '-o'); hold on; plot(tt, B.nMatch, '-o');
% ylabel('N matches'); xlabel('interval t (t \rightarrow t+1)'); grid on;
% legend('Centroid','Quadratic','Location','best');
% 更改
subplot(2,1,2);
plot(tt, A.nMatch_raw,  '-'); hold on;
plot(tt, A.nMatch_keep, '--');
plot(tt, B.nMatch_raw,  '-');
plot(tt, B.nMatch_keep, '--');
ylabel('N matches'); xlabel('interval t (t \rightarrow t+1)'); grid on;
legend('Centroid raw','Centroid keep','Quadratic raw','Quadratic keep','Location','best');

out.params = p;
out.methodA = A;
out.methodB = B;

end

% ============================ Helper functions ============================

function J = gaussBlur2D(I, sigma)
% Separable Gaussian blur without toolboxes
ks = max(7, 2*ceil(3*sigma)+1);
r = floor(ks/2);
x = (-r:r)';
g = exp(-(x.^2)/(2*sigma^2));
g = g / sum(g);
J = conv2(conv2(I, g, 'same'), g', 'same');
end

function s = robustSigma(x)
% 1.4826 * MAD around median (no toolbox)
x = x(:);
med = median(x);
s = 1.4826 * median(abs(x - med));
end

function m = localMax8(A)
% 8-neighborhood local maxima (no toolbox). Plateaus handled conservatively.
A = double(A);
m = true(size(A));
shifts = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
for k = 1:size(shifts,1)
    B = circshift(A, shifts(k,:));
    m = m & (A >= B);
end
end

function d = parabolicOffset(fm, f0, fp)
% Vertex offset for parabola through (-1,fm), (0,f0), (1,fp)
den = (fm - 2*f0 + fp);
if abs(den) < 1e-12
    d = 0;
else
    d = 0.5 * (fm - fp) / den;
end
% clamp to reasonable range
d = max(min(d, 0.5), -0.5);
end

function [dx, dy] = matchAndDisplacement(P1, P2, maxDist)
% Mutual nearest-neighbor matching (no toolbox), then displacement vectors
% P1, P2: [x y] lists (double), 1-indexed coordinates

dx = []; dy = [];
if isempty(P1) || isempty(P2), return; end

x1 = P1(:,1); y1 = P1(:,2);
x2 = P2(:,1); y2 = P2(:,2);

% squared distance matrix (n1 x n2)
D = (x1 - x2').^2 + (y1 - y2').^2;

% nearest from P1 to P2
[dn1, j12] = min(D, [], 2);
% nearest from P2 to P1
[dn2, i21] = min(D, [], 1);

% mutual pairs
i = (1:numel(j12))';
j = j12;
isMutual = (i21(j)' == i);
isClose  = (dn1 <= maxDist^2);
keep = isMutual & isClose;

i = i(keep); j = j(keep);
dx = x2(j) - x1(i);
dy = y2(j) - y1(i);
end

%更改
function [dx_med, dy_med, sigx, sigy, nRaw, nKeep] = driftAndSigma(dx, dy, resMax)
% Drift = median(dx/dy)
% Scatter = robust sigma of residuals AFTER fixed-threshold outlier removal
% nRaw  = #matches before removal
% nKeep = #matches after removal

nRaw = numel(dx);
nKeep = 0;

if nRaw < 5
    dx_med = nan; dy_med = nan; sigx = nan; sigy = nan;
    return;
end

dx_med = median(dx);
dy_med = median(dy);

rx = dx - dx_med;
ry = dy - dy_med;

% Fixed 2D residual threshold (fair across methods)
keep = hypot(rx, ry) <= resMax;

rx = rx(keep);
ry = ry(keep);
nKeep = numel(rx);

if nKeep < 5
    sigx = nan; sigy = nan;
    return;
end

sigx = robustSigma(rx);
sigy = robustSigma(ry);
end

% function [dx_med, dy_med, sigx, sigy, n] = driftAndSigma(dx, dy, outlierMAD)
% % Drift = median(dx/dy), residual scatter = robust sigma after outlier removal
% n = numel(dx);
% if n < 5
%     dx_med = nan; dy_med = nan; sigx = nan; sigy = nan;
%     return;
% end
% 
% dx_med = median(dx);
% dy_med = median(dy);
% 
% rx = dx - dx_med;
% ry = dy - dy_med;
% 
% sx = robustSigma(rx);
% sy = robustSigma(ry);
% 
% % outlier removal (optional, improves stability)
% keep = (abs(rx) <= outlierMAD*max(sx,1e-12)) & (abs(ry) <= outlierMAD*max(sy,1e-12));
% rx = rx(keep); ry = ry(keep);
% n  = numel(rx);
% 
% sigx = robustSigma(rx);
% sigy = robustSigma(ry);
% end