%% Fig. 5(a): Average Sum-Rate versus Beam Width
% Edit the parameters in this first section, then run the entire file.

clear;
clc;
close all;

%% Simulation parameters
seed = 20260703;
numTrials = 1000;                 % Use 10 or 20 for a quick test
checkpointEvery = 50;
transmitPowerDbm = 0;             % Equal transmit power of every user
noisePowerDbm = -90;              % Noise power sigma^2

M = 4;                            % Number of waveguides/receive branches
K = 4;                            % Number of single-antenna users
carrierFrequency = 28e9;          % Hz
lightSpeed = 299792458;           % m/s
height = 3;                       % Waveguide/array height in meters
regionLengthX = 30;               % D_x in meters
regionWidthY = 10;                % D_y in meters

candidateSpacing = 5;             % PA spacing in meters; L = 7
effectiveIndex = 1.4;
waveguideAttenuationDbPerM = 0.08;
beamWidths = 1:16;

runValidation = true;             % Small BnB-versus-exhaustive test
resumeFromCheckpoint = true;
scriptDirectory = fileparts(mfilename('fullpath'));
outputDirectory = fullfile(scriptDirectory, 'results');

%% Package parameters
cfg.seed = seed;
cfg.codeVersion = 6;
cfg.numTrials = numTrials;
cfg.checkpointEvery = checkpointEvery;
cfg.transmitPowerDbm = transmitPowerDbm;
cfg.noisePowerDbm = noisePowerDbm;
cfg.numWaveguides = M;
cfg.numUsers = K;
cfg.carrierFrequency = carrierFrequency;
cfg.lightSpeed = lightSpeed;
cfg.height = height;
cfg.regionLengthX = regionLengthX;
cfg.regionWidthY = regionWidthY;
cfg.candidateSpacing = candidateSpacing;
cfg.effectiveIndex = effectiveIndex;
cfg.waveguideAttenuationDbPerM = waveguideAttenuationDbPerM;
cfg.beamWidths = beamWidths;
cfg.outputDirectory = outputDirectory;
cfg.checkpointFile = fullfile(outputDirectory, 'fig5a_checkpoint.mat');
cfg.resultFile = fullfile(outputDirectory, ...
    'fig5a_sum_rate_beam_width.mat');
cfg.figureBaseName = fullfile(outputDirectory, ...
    'Fig5a_Sum_Rate_vs_Beam_Width');

assert(numTrials >= 1 && numTrials == floor(numTrials));
assert(checkpointEvery >= 1 && checkpointEvery == floor(checkpointEvery));
assert(M >= 1 && K >= 1 && M == floor(M) && K == floor(K));
assert(candidateSpacing > 0 && regionLengthX > 0 && regionWidthY > 0);
assert(all(beamWidths >= 1) && all(beamWidths == floor(beamWidths)));
assert(isscalar(noisePowerDbm) && all(isfinite(transmitPowerDbm)));
if ~exist(outputDirectory, 'dir')
    mkdir(outputDirectory);
end
if runValidation
    validate_searches(cfg);
end

%% Monte Carlo simulation
rng(seed, 'twister');
numBeams = numel(beamWidths);
noisePowerW = 10^((noisePowerDbm - 30) / 10);
transmitPowerW = 10^((transmitPowerDbm - 30) / 10);
powerNoiseRatio = transmitPowerW / noisePowerW;
sumGs = 0;
sumBs = zeros(1, numBeams);
sumBnb = 0;
sumVisited = 0;
sumSqGs = 0;
sumSqBs = zeros(1, numBeams);
sumSqBnb = 0;
startTrial = 1;

if resumeFromCheckpoint && exist(cfg.checkpointFile, 'file')
    checkpoint = load(cfg.checkpointFile);
    compatible = isfield(checkpoint, 'cfgSaved') && ...
        isequaln(checkpoint.cfgSaved, cfg);
    if compatible
        startTrial = checkpoint.completedTrials + 1;
        sumGs = checkpoint.sumGs;
        sumBs = checkpoint.sumBs;
        sumBnb = checkpoint.sumBnb;
        sumVisited = checkpoint.sumVisited;
        sumSqGs = checkpoint.sumSqGs;
        sumSqBs = checkpoint.sumSqBs;
        sumSqBnb = checkpoint.sumSqBnb;
        rng(checkpoint.randomState);
        fprintf('Resuming from trial %d.\n', startTrial);
    end
end

simulationTimer = tic;
for trial = startTrial:numTrials
    [candidateChannels, ~] = generate_channels_local(cfg);
    [~, gsRate] = greedy_search_local(candidateChannels, powerNoiseRatio);
    sumGs = sumGs + gsRate;
    sumSqGs = sumSqGs + gsRate^2;

    for beamIndex = 1:numBeams
        [~, bsRate] = beam_search_local(candidateChannels, ...
            powerNoiseRatio, beamWidths(beamIndex));
        sumBs(beamIndex) = sumBs(beamIndex) + bsRate;
        sumSqBs(beamIndex) = sumSqBs(beamIndex) + bsRate^2;
    end

    [~, bnbRate, visitedNodes] = ...
        bnb_search_local(candidateChannels, powerNoiseRatio);
    sumBnb = sumBnb + bnbRate;
    sumSqBnb = sumSqBnb + bnbRate^2;
    sumVisited = sumVisited + visitedNodes;

    if mod(trial, checkpointEvery) == 0 || trial == numTrials
        completedTrials = trial;
        cfgSaved = cfg;
        randomState = rng;
        save(cfg.checkpointFile, 'completedTrials', 'cfgSaved', ...
            'randomState', 'sumGs', 'sumBs', ...
            'sumBnb', 'sumVisited', 'sumSqGs', ...
            'sumSqBs', 'sumSqBnb', '-v7.3');
        elapsedSeconds = toc(simulationTimer);
        trialsThisRun = trial - startTrial + 1;
        remainingMinutes = (numTrials - trial) * ...
            elapsedSeconds / max(trialsThisRun, 1) / 60;
        fprintf(['Completed %d/%d trials (%.1f min elapsed, ', ...
            'approximately %.1f min remaining).\n'], ...
            trial, numTrials, elapsedSeconds / 60, remainingMinutes);
    end
end

results.gs = sumGs / numTrials;
results.bs = sumBs / numTrials;
results.bnb = sumBnb / numTrials;
results.averageBnbVisitedNodes = sumVisited / numTrials;
results.standardError.gs = ...
    standard_error_local(sumGs, sumSqGs, numTrials);
results.standardError.bs = ...
    standard_error_local(sumBs, sumSqBs, numTrials);
results.standardError.bnb = ...
    standard_error_local(sumBnb, sumSqBnb, numTrials);
results.transmitPowerDbm = transmitPowerDbm;
results.noisePowerDbm = noisePowerDbm;
results.powerNoiseRatioDb = transmitPowerDbm - noisePowerDbm;
results.beamWidths = beamWidths;
results.relativeGapToBnb = (results.bnb - results.bs) / results.bnb;
index99 = find(results.bs >= 0.99 * results.bnb, 1, 'first');
if isempty(index99)
    results.minimumBeamWidthFor99Percent = NaN;
else
    results.minimumBeamWidthFor99Percent = beamWidths(index99);
end
results.config = cfg;
save(cfg.resultFile, 'results', 'cfg', '-v7.3');

%% Plot and export Fig. 5(a)
figureHandle = figure('Color', 'w', 'Position', [100, 100, 760, 570]);
hold on;
plot(beamWidths, results.gs * ones(size(beamWidths)), '--s', ...
    'Color', [0.00, 0.45, 0.74], ...
    'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', 'w');
plot(beamWidths, results.bs, '-o', 'Color', [0.85, 0.33, 0.10], ...
    'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', 'w');
plot(beamWidths, results.bnb * ones(size(beamWidths)), '-.d', ...
    'Color', [0.49, 0.18, 0.56], ...
    'LineWidth', 1.8, 'MarkerSize', 8, 'MarkerFaceColor', 'w');

legendEntries = {'GS', 'BS', 'BnB'};
xlabel('Beam Width B');
ylabel('Average Sum-Rate [bps/Hz]');
legend(legendEntries, 'Location', 'southeast', 'Box', 'off');
grid on;
box on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13, ...
    'LineWidth', 1.1, 'XMinorTick', 'on', 'YMinorTick', 'on');
xlim([beamWidths(1), beamWidths(end)]);
xticks(beamWidths);

savefig(figureHandle, [cfg.figureBaseName, '.fig']);
print(figureHandle, [cfg.figureBaseName, '.eps'], '-depsc', '-r600');
print(figureHandle, [cfg.figureBaseName, '.pdf'], '-dpdf', '-r600');
fprintf('Results saved in %s\n', outputDirectory);

%% Local functions
function [candidateChannels, fixedChannel] = generate_channels_local(cfg)
M = cfg.numWaveguides;
K = cfg.numUsers;
lambda = cfg.lightSpeed / cfg.carrierFrequency;
k0 = 2 * pi / lambda;
kg = cfg.effectiveIndex * k0;
etaSqrt = cfg.lightSpeed / (4 * pi * cfg.carrierFrequency);
userX = cfg.regionLengthX * rand(1, K);
userY = cfg.regionWidthY * (rand(1, K) - 0.5);
waveguideY = linspace(-cfg.regionWidthY / 2, ...
    cfg.regionWidthY / 2, M);
candidateX = 0:cfg.candidateSpacing:cfg.regionLengthX;
if abs(candidateX(end) - cfg.regionLengthX) > 10 * eps
    candidateX = [candidateX, cfg.regionLengthX];
end
L = numel(candidateX);

candidateChannels = complex(zeros(M, L, K));
guideLoss = 10.^(-cfg.waveguideAttenuationDbPerM * candidateX / 20);
guidePhase = exp(-1i * kg * candidateX);
for m = 1:M
    for k = 1:K
        distance = sqrt((candidateX - userX(k)).^2 + ...
            (waveguideY(m) - userY(k)).^2 + cfg.height^2);
        freeSpaceChannel = etaSqrt * exp(-1i * k0 * distance) ./ distance;
        candidateChannels(m, :, k) = ...
            guideLoss .* guidePhase .* freeSpaceChannel;
    end
end

fixedX = (cfg.regionLengthX / 2) * ones(1, M);
fixedY = ((0:M-1) - (M-1)/2) * (lambda / 2);
fixedChannel = complex(zeros(M, K));
for m = 1:M
    for k = 1:K
        distance = sqrt((fixedX(m) - userX(k))^2 + ...
            (fixedY(m) - userY(k))^2 + cfg.height^2);
        fixedChannel(m, k) = ...
            etaSqrt * exp(-1i * k0 * distance) / distance;
    end
end
end

function rate = sum_rate_local(channel, snrLinear)
K = size(channel, 2);
A = eye(K) + snrLinear * (channel' * channel);
A = (A + A') / 2;
[R, flag] = chol(A);
if flag == 0
    rate = 2 * sum(log2(real(diag(R))));
else
    rate = sum(log2(max(real(eig(A)), eps)));
end
end

function [indices, rate, inverseMatrix] = ...
    greedy_search_local(candidateChannels, snrLinear)
[M, L, K] = size(candidateChannels);
indices = zeros(1, M);
rate = 0;
inverseMatrix = eye(K);
for m = 1:M
    increments = zeros(1, L);
    for ell = 1:L
        row = sqrt(snrLinear) * ...
            reshape(candidateChannels(m, ell, :), 1, K);
        gain = max(real(row * inverseMatrix * row'), 0);
        increments(ell) = log2(1 + gain);
    end
    [bestIncrement, indices(m)] = max(increments);
    row = sqrt(snrLinear) * ...
        reshape(candidateChannels(m, indices(m), :), 1, K);
    denominator = 1 + real(row * inverseMatrix * row');
    inverseMatrix = inverseMatrix - ...
        (inverseMatrix * (row' * row) * inverseMatrix) / denominator;
    inverseMatrix = (inverseMatrix + inverseMatrix') / 2;
    rate = rate + bestIncrement;
end
end

function [bestIndices, bestRate] = ...
    beam_search_local(candidateChannels, snrLinear, beamWidth)
[M, L, K] = size(candidateChannels);
paths = zeros(1, 0);
rates = 0;
inverseMatrices = reshape(eye(K), K, K, 1);
for m = 1:M
    numParents = size(paths, 1);
    numChildren = numParents * L;
    childPaths = zeros(numChildren, m);
    childRates = zeros(numChildren, 1);
    childInverses = complex(zeros(K, K, numChildren));
    childIndex = 0;
    for p = 1:numParents
        parentInverse = inverseMatrices(:, :, p);
        for ell = 1:L
            childIndex = childIndex + 1;
            row = sqrt(snrLinear) * ...
                reshape(candidateChannels(m, ell, :), 1, K);
            gain = max(real(row * parentInverse * row'), 0);
            denominator = 1 + gain;
            childPaths(childIndex, :) = [paths(p, :), ell];
            childRates(childIndex) = rates(p) + log2(denominator);
            updatedInverse = parentInverse - ...
                (parentInverse * (row' * row) * parentInverse) / denominator;
            childInverses(:, :, childIndex) = ...
                (updatedInverse + updatedInverse') / 2;
        end
    end
    keepCount = min(beamWidth, numChildren);
    [~, order] = sort(childRates, 'descend');
    keep = order(1:keepCount);
    paths = childPaths(keep, :);
    rates = childRates(keep);
    inverseMatrices = childInverses(:, :, keep);
end
[bestRate, best] = max(rates);
bestIndices = paths(best, :);
end

function [bestIndices, bestRate, visitedNodes] = ...
    bnb_search_local(candidateChannels, snrLinear)
[M, L, K] = size(candidateChannels);
effectiveChannels = sqrt(snrLinear) .* candidateChannels;
Z = zeros(1, M);
for m = 1:M
    normsSquared = zeros(1, L);
    for candidateIndex = 1:L
        row = reshape(effectiveChannels(m, candidateIndex, :), 1, K);
        normsSquared(candidateIndex) = real(row * row');
    end
    Z(m) = log2(1 + max(normsSquared));
end
offsetPrefix = [0, cumsum(Z)];
totalOffset = offsetPrefix(end);
[greedyIndices, greedyRate] = ...
    greedy_search_local(candidateChannels, snrLinear);
bestIndices = greedyIndices;
bestRate = greedyRate;
incumbent = greedyRate - totalOffset;
visitedNodes = 0;
workingIndices = zeros(1, M);
search_node(1, 0, eye(K));

    function search_node(layer, partialRate, inverseMatrix)
        increments = zeros(1, L);
        rows = complex(zeros(L, K));
        for candidate = 1:L
            rows(candidate, :) = reshape( ...
                effectiveChannels(layer, candidate, :), 1, K);
            gain = max(real(rows(candidate, :) * inverseMatrix * ...
                rows(candidate, :)'), 0);
            increments(candidate) = log2(1 + gain);
        end
        [~, order] = sort(increments, 'descend');
        for orderIndex = 1:L
            ell = order(orderIndex);
            visitedNodes = visitedNodes + 1;
            childRate = partialRate + increments(ell);
            transformedRate = childRate - offsetPrefix(layer + 1);
            if transformedRate <= incumbent + 1e-12
                continue;
            end
            workingIndices(layer) = ell;
            if layer == M
                incumbent = transformedRate;
                bestRate = childRate;
                bestIndices = workingIndices;
                continue;
            end
            row = rows(ell, :);
            denominator = 1 + real(row * inverseMatrix * row');
            childInverse = inverseMatrix - ...
                (inverseMatrix * (row' * row) * inverseMatrix) / denominator;
            childInverse = (childInverse + childInverse') / 2;
            search_node(layer + 1, childRate, childInverse);
        end
    end
end

function validate_searches(cfg)
testCfg = cfg;
testCfg.numWaveguides = 3;
testCfg.numUsers = 3;
testCfg.regionLengthX = 4;
testCfg.regionWidthY = 4;
testCfg.candidateSpacing = 1;
savedState = rng;
rng(17, 'twister');
validationPowerRatioDb = [70, 85, 100];
for trial = 1:3
    [channels, ~] = generate_channels_local(testCfg);
    for powerIndex = 1:numel(validationPowerRatioDb)
        powerNoiseRatio = 10^(validationPowerRatioDb(powerIndex) / 10);
        [~, gsRate] = greedy_search_local(channels, powerNoiseRatio);
        [~, bsOneRate] = beam_search_local(channels, powerNoiseRatio, 1);
        [~, bsTwoRate] = beam_search_local(channels, powerNoiseRatio, 2);
        [~, bsFourRate] = beam_search_local(channels, powerNoiseRatio, 4);
        [bnbIndices, bnbRate] = bnb_search_local(channels, powerNoiseRatio);
        exhaustiveRate = exhaustive_search_local(channels, powerNoiseRatio);
        selectedBnbChannel = complex(zeros(testCfg.numWaveguides, ...
            testCfg.numUsers));
        for m = 1:testCfg.numWaveguides
            selectedBnbChannel(m, :) = reshape( ...
                channels(m, bnbIndices(m), :), 1, testCfg.numUsers);
        end
        directBnbRate = sum_rate_local(selectedBnbChannel, powerNoiseRatio);
        assert(abs(gsRate - bsOneRate) < 1e-9, ...
            'Validation failed: BS with B=1 differs from GS.');
        assert(abs(bnbRate - exhaustiveRate) < 1e-8, ...
            'Validation failed: BnB differs from exhaustive search.');
        assert(abs(bnbRate - directBnbRate) < 1e-8, ...
            'Validation failed: recursive and direct BnB rates differ.');
        assert(bnbRate + 1e-10 >= max([gsRate, bsTwoRate, bsFourRate]), ...
            'Validation failed: a heuristic exceeds the optimum.');
    end
end
rng(savedState);
fprintf('Search validation passed.\n');
end

function standardError = standard_error_local(sumValues, sumSquares, count)
if count <= 1
    standardError = zeros(size(sumValues));
    return;
end
sampleVariance = (sumSquares - (sumValues.^2) / count) / (count - 1);
sampleVariance = max(real(sampleVariance), 0);
standardError = sqrt(sampleVariance / count);
end

function bestRate = exhaustive_search_local(channels, snrLinear)
[M, L, K] = size(channels);
indices = ones(1, M);
bestRate = -inf;
while true
    selected = complex(zeros(M, K));
    for m = 1:M
        selected(m, :) = reshape(channels(m, indices(m), :), 1, K);
    end
    bestRate = max(bestRate, sum_rate_local(selected, snrLinear));
    position = M;
    while position >= 1 && indices(position) == L
        indices(position) = 1;
        position = position - 1;
    end
    if position == 0
        break;
    end
    indices(position) = indices(position) + 1;
end
end
