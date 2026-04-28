function manifest = generate_click_wavs(outDir, freqsHz, durationSec, fs, clickMs, amp, seed)
%GENERATE_CLICK_WAVS Generate periodic and random click WAV files.
%   manifest = generate_click_wavs(outDir, freqsHz, durationSec, fs, clickMs, amp, seed)
%
% Example:
%   generate_click_wavs('outputs/stimuli_matlab', [2 4 6 8 10 12], 120, 44100, 5, 0.75, 42);

if nargin < 1 || isempty(outDir), outDir = fullfile('outputs', 'stimuli_matlab'); end
if nargin < 2 || isempty(freqsHz), freqsHz = [2 4 6 8 10 12]; end
if nargin < 3 || isempty(durationSec), durationSec = 120; end
if nargin < 4 || isempty(fs), fs = 44100; end
if nargin < 5 || isempty(clickMs), clickMs = 5; end
if nargin < 6 || isempty(amp), amp = 0.75; end
if nargin < 7 || isempty(seed), seed = 42; end

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

rng(seed);
nSamples = round(durationSec * fs);
clickSamples = max(1, round(clickMs * 1e-3 * fs));
clickKernel = amp * (2 * rand(clickSamples, 1) - 1) .* hann(clickSamples);

manifest = table('Size', [0 6], ...
    'VariableTypes', {'string','string','double','double','double','double'}, ...
    'VariableNames', {'stimulus_name','path','frequency_hz','is_random','duration_s','sample_rate_hz'});

for i = 1:numel(freqsHz)
    hz = freqsHz(i);

    periodic = make_periodic(nSamples, fs, hz, clickKernel);
    periodicName = sprintf('periodic_%dhz.wav', round(hz));
    periodicPath = fullfile(outDir, periodicName);
    audiowrite(periodicPath, clamp_audio(periodic), fs);
    manifest = [manifest; {string(periodicName), string(periodicPath), hz, 0, durationSec, fs}]; %#ok<AGROW>

    randomSig = make_poisson(nSamples, fs, hz, clickKernel);
    randomName = sprintf('random_%dhz.wav', round(hz));
    randomPath = fullfile(outDir, randomName);
    audiowrite(randomPath, clamp_audio(randomSig), fs);
    manifest = [manifest; {string(randomName), string(randomPath), hz, 1, durationSec, fs}]; %#ok<AGROW>
end

silence = zeros(nSamples, 1);
silenceName = "silence.wav";
silencePath = fullfile(outDir, silenceName);
audiowrite(silencePath, silence, fs);
manifest = [manifest; {silenceName, string(silencePath), NaN, NaN, durationSec, fs}];

manifestPath = fullfile(outDir, 'stimulus_manifest.csv');
writetable(manifest, manifestPath);
fprintf('Wrote WAVs to: %s\n', outDir);
fprintf('Wrote manifest: %s\n', manifestPath);
end

function y = make_periodic(nSamples, fs, hz, clickKernel)
y = zeros(nSamples, 1);
step = max(1, round(fs / hz));
for idx = 1:step:nSamples
    j = min(nSamples, idx + numel(clickKernel) - 1);
    y(idx:j) = y(idx:j) + clickKernel(1:(j - idx + 1));
end
end

function y = make_poisson(nSamples, fs, rateHz, clickKernel)
y = zeros(nSamples, 1);
t = 0;
totalSec = nSamples / fs;
while t < totalSec
    t = t + exprnd(1 / rateHz);
    idx = floor(t * fs) + 1;
    if idx > nSamples
        break;
    end
    j = min(nSamples, idx + numel(clickKernel) - 1);
    y(idx:j) = y(idx:j) + clickKernel(1:(j - idx + 1));
end
end

function y = clamp_audio(y)
y = max(min(y, 1.0), -1.0);
end
