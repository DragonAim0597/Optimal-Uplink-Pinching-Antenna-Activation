%% Fig. 4: Sum-Rate and Runtime versus Number of Candidates L
% Edit the parameters in this first section, then run the entire file.

clear;
clc;
close all;

%% Simulation parameters
seed = 20260703;
numTrials = 1000;
checkpointEvery = 20;
transmitPowerDbm = 0;             % Equal transmit power of every user
noisePowerDbm = -90;              % Noise power sigma^2

M = 4;                            % Number of waveguides/receive branches
K = 4;                            % Number of single-antenna users
carrierFrequency = 28e9;          % Hz
lightSpeed = 299792458;           % m/s
height = 3;                       % Waveguide/array height in meters
regionLengthX = 30;               % D_x in meters
regionWidthY = 10;                % D_y in meters

candidateSpacingValues = [10, 5, 2, 1, 0.5, 0.2];
numCandidateValues = round(regionLengthX ./ candidateSpacingValues) + 1;
effectiveIndex = 1.4;
waveguideAttenuationDbPerM = 0.08;
beamWidths = [2, 4, 8];

runValidation = true;             % Small BnB-versus-exhaustive test
resumeFromCheckpoint = true;
useParallel = false;              % Serial Monte Carlo execution
numWorkers = [];                  % [] lets MATLAB choose the pool size
scriptDirectory = fileparts(mfilename('fullpath'));
outputDirectory = fullfile(scriptDirectory, 'results');

%% Package parameters
cfg.seed = seed;
cfg.codeVersion = 17;
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
cfg.candidateSpacingValues = candidateSpacingValues;
cfg.numCandidateValues = numCandidateValues;
cfg.numCandidates = numCandidateValues(1);
cfg.effectiveIndex = effectiveIndex;
cfg.waveguideAttenuationDbPerM = waveguideAttenuationDbPerM;
cfg.beamWidths = beamWidths;
cfg.useParallel = useParallel;
cfg.numWorkers = numWorkers;
cfg.outputDirectory = outputDirectory;
cfg.checkpointFile = fullfile(outputDirectory, ...
    'fig4_L_checkpoint.mat');
cfg.resultFile = fullfile(outputDirectory, ...
    'fig4_performance_runtime_L.mat');
cfg.figureBaseName = fullfile(outputDirectory, ...
    'Fig4a_Sum_Rate_vs_Number_of_Candidates');

assert(numTrials >= 1 && numTrials == floor(numTrials));
assert(checkpointEvery >= 1 && checkpointEvery == floor(checkpointEvery));
assert(M >= 1 && K >= 1 && M == floor(M) && K == floor(K));
assert(all(numCandidateValues >= 2) && ...
    all(numCandidateValues == floor(numCandidateValues)) && ...
    regionLengthX > 0 && ...
    regionWidthY > 0);
assert(max(abs(regionLengthX ./ (numCandidateValues - 1) - ...
    candidateSpacingValues)) < 1e-12);
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
numLPoints = numel(numCandidateValues);
numBeams = numel(beamWidths);
noisePowerW = 10^((noisePowerDbm - 30) / 10);
transmitPowerW = 10^((transmitPowerDbm - 30) / 10);
powerNoiseRatio = transmitPowerW / noisePowerW;
sumFixed = zeros(1, numLPoints);
sumGs = zeros(1, numLPoints);
sumBs = zeros(numBeams, numLPoints);
sumBnb = zeros(1, numLPoints);
sumVisited = zeros(1, numLPoints);
sumSqFixed = zeros(1, numLPoints);
sumSqGs = zeros(1, numLPoints);
sumSqBs = zeros(numBeams, numLPoints);
sumSqBnb = zeros(1, numLPoints);
sumTimeGs = zeros(1, numLPoints);
sumTimeBs = zeros(numBeams, numLPoints);
sumTimeBnb = zeros(1, numLPoints);
sumSqTimeGs = zeros(1, numLPoints);
sumSqTimeBs = zeros(numBeams, numLPoints);
sumSqTimeBnb = zeros(1, numLPoints);
startTrial = 1;

if resumeFromCheckpoint && exist(cfg.checkpointFile, 'file')
    checkpoint = load(cfg.checkpointFile);
    compatible = isfield(checkpoint, 'cfgSaved') && ...
        isequaln(checkpoint.cfgSaved, cfg);
    if compatible
        startTrial = checkpoint.completedTrials + 1;
        sumFixed = checkpoint.sumFixed;
        sumGs = checkpoint.sumGs;
        sumBs = checkpoint.sumBs;
        sumBnb = checkpoint.sumBnb;
        sumVisited = checkpoint.sumVisited;
        sumSqFixed = checkpoint.sumSqFixed;
        sumSqGs = checkpoint.sumSqGs;
        sumSqBs = checkpoint.sumSqBs;
        sumSqBnb = checkpoint.sumSqBnb;
        sumTimeGs = checkpoint.sumTimeGs;
        sumTimeBs = checkpoint.sumTimeBs;
        sumTimeBnb = checkpoint.sumTimeBnb;
        sumSqTimeGs = checkpoint.sumSqTimeGs;
        sumSqTimeBs = checkpoint.sumSqTimeBs;
        sumSqTimeBnb = checkpoint.sumSqTimeBnb;
        fprintf('Resuming from trial %d.\n', startTrial);
    end
end

parallelAvailable = license('test', 'Distrib_Computing_Toolbox');
runInParallel = useParallel && parallelAvailable;
if useParallel && ~parallelAvailable
    warning('Parallel Computing Toolbox is unavailable. Using serial batches.');
end
if runInParallel
    pool = gcp('nocreate');
    if isempty(pool)
        if isempty(numWorkers)
            pool = parpool('local');
        else
            pool = parpool('local', numWorkers);
        end
    end
    fprintf('Using %d parallel workers.\n', pool.NumWorkers);
end

simulationTimer = tic;
for batchStart = startTrial:checkpointEvery:numTrials
    trialIds = batchStart:min(batchStart + checkpointEvery - 1, numTrials);
    batchCount = numel(trialIds);
    batchFixed = zeros(batchCount, numLPoints);
    batchGs = zeros(batchCount, numLPoints);
    batchBs = zeros(batchCount, numBeams, numLPoints);
    batchBnb = zeros(batchCount, numLPoints);
    batchVisited = zeros(batchCount, numLPoints);
    batchTimeGs = zeros(batchCount, numLPoints);
    batchTimeBs = zeros(batchCount, numBeams, numLPoints);
    batchTimeBnb = zeros(batchCount, numLPoints);

    if runInParallel
        parfor batchIndex = 1:batchCount
            trialSeed = seed + trialIds(batchIndex);
            [fixedRow, gsRow, bsRows, bnbRow, visitedRow, ...
                timeGsRow, timeBsRows, timeBnbRow] = run_trial_local( ...
                cfg, numCandidateValues, beamWidths, ...
                powerNoiseRatio, trialSeed);
            batchFixed(batchIndex, :) = fixedRow;
            batchGs(batchIndex, :) = gsRow;
            batchBs(batchIndex, :, :) = ...
                reshape(bsRows, 1, numBeams, numLPoints);
            batchBnb(batchIndex, :) = bnbRow;
            batchVisited(batchIndex, :) = visitedRow;
            batchTimeGs(batchIndex, :) = timeGsRow;
            batchTimeBs(batchIndex, :, :) = ...
                reshape(timeBsRows, 1, numBeams, numLPoints);
            batchTimeBnb(batchIndex, :) = timeBnbRow;
        end
    else
        for batchIndex = 1:batchCount
            trialSeed = seed + trialIds(batchIndex);
            [fixedRow, gsRow, bsRows, bnbRow, visitedRow, ...
                timeGsRow, timeBsRows, timeBnbRow] = run_trial_local( ...
                cfg, numCandidateValues, beamWidths, ...
                powerNoiseRatio, trialSeed);
            batchFixed(batchIndex, :) = fixedRow;
            batchGs(batchIndex, :) = gsRow;
            batchBs(batchIndex, :, :) = ...
                reshape(bsRows, 1, numBeams, numLPoints);
            batchBnb(batchIndex, :) = bnbRow;
            batchVisited(batchIndex, :) = visitedRow;
            batchTimeGs(batchIndex, :) = timeGsRow;
            batchTimeBs(batchIndex, :, :) = ...
                reshape(timeBsRows, 1, numBeams, numLPoints);
            batchTimeBnb(batchIndex, :) = timeBnbRow;
        end
    end

    sumFixed = sumFixed + sum(batchFixed, 1);
    sumGs = sumGs + sum(batchGs, 1);
    sumBs = sumBs + reshape(sum(batchBs, 1), numBeams, numLPoints);
    sumBnb = sumBnb + sum(batchBnb, 1);
    sumVisited = sumVisited + sum(batchVisited, 1);
    sumSqFixed = sumSqFixed + sum(batchFixed.^2, 1);
    sumSqGs = sumSqGs + sum(batchGs.^2, 1);
    sumSqBs = sumSqBs + ...
        reshape(sum(batchBs.^2, 1), numBeams, numLPoints);
    sumSqBnb = sumSqBnb + sum(batchBnb.^2, 1);
    sumTimeGs = sumTimeGs + sum(batchTimeGs, 1);
    sumTimeBs = sumTimeBs + ...
        reshape(sum(batchTimeBs, 1), numBeams, numLPoints);
    sumTimeBnb = sumTimeBnb + sum(batchTimeBnb, 1);
    sumSqTimeGs = sumSqTimeGs + sum(batchTimeGs.^2, 1);
    sumSqTimeBs = sumSqTimeBs + ...
        reshape(sum(batchTimeBs.^2, 1), numBeams, numLPoints);
    sumSqTimeBnb = sumSqTimeBnb + sum(batchTimeBnb.^2, 1);

    completedTrials = trialIds(end);
    cfgSaved = cfg;
    save(cfg.checkpointFile, 'completedTrials', 'cfgSaved', ...
        'sumFixed', 'sumGs', 'sumBs', 'sumBnb', 'sumVisited', ...
        'sumSqFixed', 'sumSqGs', 'sumSqBs', 'sumSqBnb', ...
        'sumTimeGs', 'sumTimeBs', 'sumTimeBnb', 'sumSqTimeGs', ...
        'sumSqTimeBs', 'sumSqTimeBnb', '-v7.3');
    elapsedSeconds = toc(simulationTimer);
    trialsThisRun = completedTrials - startTrial + 1;
    remainingMinutes = (numTrials - completedTrials) * ...
        elapsedSeconds / max(trialsThisRun, 1) / 60;
    fprintf(['Completed %d/%d trials (%.1f min elapsed, ', ...
        'approximately %.1f min remaining).\n'], ...
        completedTrials, numTrials, elapsedSeconds / 60, remainingMinutes);
end

results.fixed = sumFixed / numTrials;
results.gs = sumGs / numTrials;
results.bs = sumBs / numTrials;
results.bnb = sumBnb / numTrials;
results.averageBnbVisitedNodes = sumVisited / numTrials;
results.standardError.fixed = ...
    standard_error_local(sumFixed, sumSqFixed, numTrials);
results.standardError.gs = ...
    standard_error_local(sumGs, sumSqGs, numTrials);
results.standardError.bs = ...
    standard_error_local(sumBs, sumSqBs, numTrials);
results.standardError.bnb = ...
    standard_error_local(sumBnb, sumSqBnb, numTrials);
results.numCandidates = numCandidateValues;
results.candidateSpacing = candidateSpacingValues;
results.runtimeMs.gs = sumTimeGs / numTrials;
results.runtimeMs.bs = sumTimeBs / numTrials;
results.runtimeMs.bnb = sumTimeBnb / numTrials;
results.runtimeStandardErrorMs.gs = ...
    standard_error_local(sumTimeGs, sumSqTimeGs, numTrials);
results.runtimeStandardErrorMs.bs = ...
    standard_error_local(sumTimeBs, sumSqTimeBs, numTrials);
results.runtimeStandardErrorMs.bnb = ...
    standard_error_local(sumTimeBnb, sumSqTimeBnb, numTrials);
results.runtimeScope = ...
    'Search algorithm only; channel generation and plotting excluded';
results.transmitPowerDbm = transmitPowerDbm;
results.noisePowerDbm = noisePowerDbm;
results.powerNoiseRatioDb = transmitPowerDbm - noisePowerDbm;
results.beamWidths = beamWidths;
results.config = cfg;
save(cfg.resultFile, 'results', 'cfg', '-v7.3');

%% Plot and export Fig. 4
figureHandle = figure('Color', 'w', 'Position', [100, 100, 760, 570]);
hold on;
colors = [0.15, 0.15, 0.15; 0.00, 0.45, 0.74; ...
          0.85, 0.33, 0.10; 0.47, 0.67, 0.19; ...
          0.49, 0.18, 0.56; 0.64, 0.08, 0.18];
numCurves = numBeams + 3;
if numCurves > size(colors, 1)
    colors = lines(numCurves);
    colors(1, :) = [0.15, 0.15, 0.15];
end
styles = {'--o', '-s', '-.^', '-.d', '-.v', '-p', '-->', '--<'};

plot(numCandidateValues, results.fixed, styles{1}, ...
    'Color', colors(1, :), ...
    'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', 'w');
plot(numCandidateValues, results.gs, styles{2}, ...
    'Color', colors(2, :), ...
    'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', 'w');
for beamIndex = 1:numBeams
    styleIndex = 1 + mod(beamIndex + 1, numel(styles));
    plot(numCandidateValues, results.bs(beamIndex, :), ...
        styles{styleIndex}, ...
        'Color', colors(2 + beamIndex, :), 'LineWidth', 1.8, ...
        'MarkerSize', 7, 'MarkerFaceColor', 'w');
end
bnbCurveIndex = numBeams + 3;
bnbStyleIndex = 1 + mod(bnbCurveIndex - 1, numel(styles));
plot(numCandidateValues, results.bnb, styles{bnbStyleIndex}, ...
    'Color', colors(bnbCurveIndex, :), ...
    'LineWidth', 1.8, 'MarkerSize', 8, 'MarkerFaceColor', 'w');

legendEntries = {'Fixed \lambda/2 array (centered)', 'GS'};
for beamIndex = 1:numBeams
    legendEntries{end + 1} = sprintf('BeS (Beam Width = %d)', ...
        beamWidths(beamIndex)); %#ok<SAGROW>
end
legendEntries{end + 1} = 'BnB';
xlabel('Number of Candidate Locations L');
ylabel('Average Sum-Rate [bps/Hz]');
legend(legendEntries, 'Location', 'northwest', 'Box', 'off');
grid on;
box on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13, ...
    'LineWidth', 1.1, 'XMinorTick', 'on', 'YMinorTick', 'on');
xlim([numCandidateValues(1), numCandidateValues(end)]);
xticks(numCandidateValues);

savefig(figureHandle, [cfg.figureBaseName, '.fig']);
print(figureHandle, [cfg.figureBaseName, '.eps'], '-depsc', '-r600');
print(figureHandle, [cfg.figureBaseName, '.pdf'], '-dpdf', '-r600');

%% Plot and export the runtime companion figure
runtimeFigure = figure('Color', 'w', 'Position', [120, 120, 760, 570]);
hold on;
plot(numCandidateValues, results.runtimeMs.gs, styles{2}, ...
    'Color', colors(2, :), 'LineWidth', 1.8, ...
    'MarkerSize', 7, 'MarkerFaceColor', 'w');
for beamIndex = 1:numBeams
    styleIndex = 1 + mod(beamIndex + 1, numel(styles));
    plot(numCandidateValues, results.runtimeMs.bs(beamIndex, :), ...
        styles{styleIndex}, 'Color', colors(2 + beamIndex, :), ...
        'LineWidth', 1.8, 'MarkerSize', 7, 'MarkerFaceColor', 'w');
end
plot(numCandidateValues, results.runtimeMs.bnb, styles{bnbStyleIndex}, ...
    'Color', colors(bnbCurveIndex, :), 'LineWidth', 1.8, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'w');
xlabel('Number of Candidate Locations L');
ylabel('Average Runtime [ms]');
legend(legendEntries(2:end), 'Location', 'northwest', 'Box', 'off');
grid on;
box on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13, ...
    'LineWidth', 1.1, 'XMinorTick', 'on', 'YMinorTick', 'on');
xlim([numCandidateValues(1), numCandidateValues(end)]);
xticks(numCandidateValues);
runtimeBaseName = fullfile(outputDirectory, ...
    'Fig4b_Runtime_vs_Number_of_Candidates');
savefig(runtimeFigure, [runtimeBaseName, '.fig']);
print(runtimeFigure, [runtimeBaseName, '.eps'], '-depsc', '-r600');
print(runtimeFigure, [runtimeBaseName, '.pdf'], '-dpdf', '-r600');
fprintf('Results saved in %s\n', outputDirectory);

%% Local functions
function [fixedRates, gsRates, bsRates, bnbRates, visitedNodes, ...
    gsTimes, bsTimes, bnbTimes] = run_trial_local( ...
    cfg, numCandidateValues, beamWidths, powerNoiseRatio, trialSeed)

rng(trialSeed, 'twister');
K = cfg.numUsers;
numLPoints = numel(numCandidateValues);
numBeams = numel(beamWidths);
normalizedUserX = rand(1, K);
normalizedUserY = rand(1, K) - 0.5;

% Compile the search paths on this worker before collecting timings.
warmupCfg = cfg;
warmupCfg.numCandidates = numCandidateValues(1);
warmupCfg.normalizedUserX = normalizedUserX;
warmupCfg.normalizedUserY = normalizedUserY;
[warmupChannels, ~] = generate_channels_local(warmupCfg);
greedy_search_local(warmupChannels, powerNoiseRatio);
for beamIndex = 1:numBeams
    beam_search_local(warmupChannels, powerNoiseRatio, ...
        beamWidths(beamIndex));
end
bnb_search_local(warmupChannels, powerNoiseRatio);

fixedRates = zeros(1, numLPoints);
gsRates = zeros(1, numLPoints);
bsRates = zeros(numBeams, numLPoints);
bnbRates = zeros(1, numLPoints);
visitedNodes = zeros(1, numLPoints);
gsTimes = zeros(1, numLPoints);
bsTimes = zeros(numBeams, numLPoints);
bnbTimes = zeros(1, numLPoints);

for lIndex = 1:numLPoints
    trialCfg = cfg;
    trialCfg.numCandidates = numCandidateValues(lIndex);
    trialCfg.normalizedUserX = normalizedUserX;
    trialCfg.normalizedUserY = normalizedUserY;
    [candidateChannels, fixedChannel] = generate_channels_local(trialCfg);
    fixedRates(lIndex) = sum_rate_local(fixedChannel, powerNoiseRatio);

    algorithmTimer = tic;
    [~, gsRates(lIndex)] = ...
        greedy_search_local(candidateChannels, powerNoiseRatio);
    gsTimes(lIndex) = 1e3 * toc(algorithmTimer);

    for beamIndex = 1:numBeams
        algorithmTimer = tic;
        [~, bsRates(beamIndex, lIndex)] = beam_search_local( ...
            candidateChannels, powerNoiseRatio, beamWidths(beamIndex));
        bsTimes(beamIndex, lIndex) = 1e3 * toc(algorithmTimer);
    end

    algorithmTimer = tic;
    [~, bnbRates(lIndex), visitedNodes(lIndex)] = ...
        bnb_search_local(candidateChannels, powerNoiseRatio);
    bnbTimes(lIndex) = 1e3 * toc(algorithmTimer);
end
end

function [candidateChannels, fixedChannel] = generate_channels_local(cfg)
M = cfg.numWaveguides;
K = cfg.numUsers;
lambda = cfg.lightSpeed / cfg.carrierFrequency;
k0 = 2 * pi / lambda;
kg = cfg.effectiveIndex * k0;
etaSqrt = cfg.lightSpeed / (4 * pi * cfg.carrierFrequency);
if isfield(cfg, 'normalizedUserX') && isfield(cfg, 'normalizedUserY')
    userX = cfg.regionLengthX * cfg.normalizedUserX;
    userY = cfg.regionWidthY * cfg.normalizedUserY;
else
    userX = cfg.regionLengthX * rand(1, K);
    userY = cfg.regionWidthY * (rand(1, K) - 0.5);
end
waveguideY = linspace(-cfg.regionWidthY / 2, ...
    cfg.regionWidthY / 2, M);
candidateX = linspace(0, cfg.regionLengthX, cfg.numCandidates);
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
testCfg.numCandidates = 5;
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
