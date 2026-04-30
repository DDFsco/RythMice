function manifest = generate_session_sequence_wavs(outDir, silenceSec, soundSec, fs, clickMs, amp, baseSeed)
%GENERATE_SESSION_SEQUENCE_WAVS Long session WAVs: 0 = silence 4min, 2/7 = Hz for 2min.
%   manifest = generate_session_sequence_wavs(...)
%
% Writes five files to outputs/stimuli_sessions (by default):
%   1,3,5: [0 2 0 7 0 2 0 7 0 2]
%   2,4:   [0 7 0 2 0 7 0 2 0 7]
%
% Seeds: repeating macro-sequences share identical waveform (templates A/B).

if nargin < 1 || isempty(outDir)
    outDir = fullfile('outputs', 'stimuli_sessions');
end
if nargin < 2 || isempty(silenceSec), silenceSec = 240; end
if nargin < 3 || isempty(soundSec), soundSec = 120; end
if nargin < 4 || isempty(fs), fs = 44100; end
if nargin < 5 || isempty(clickMs), clickMs = 5; end
if nargin < 6 || isempty(amp), amp = 0.75; end
if nargin < 7 || isempty(baseSeed), baseSeed = 42; end

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

planA = [0 2 0 7 0 2 0 7 0 2];
planB = [0 7 0 2 0 7 0 2 0 7];
plans = {planA, planB, planA, planB, planA};
seeds = baseSeed + [1000, 2000, 1000, 2000, 1000];

manifest = table('Size', [0 9], ...
    'VariableTypes', {'double','string','string','string','double','double','double','double','double'}, ...
    'VariableNames', {'session_index','filename','path','sequence_codes','silence_segment_s','sound_segment_s','total_duration_s','sample_rate_hz','rng_seed'});

for k = 1:numel(plans)
    rng(seeds(k));
    codes = plans{k};
    clickSamples = max(1, round(clickMs * 1e-3 * fs));
    clickKernel = amp * (2 * rand(clickSamples, 1) - 1) .* hann(clickSamples);

    y = [];
    for j = 1:numel(codes)
        code = codes(j);
        if code == 0
            ns = round(silenceSec * fs);
            y = [y; zeros(ns, 1)]; %#ok<AGROW>
        elseif code == 2 || code == 7
            ns = round(soundSec * fs);
            y = [y; make_periodic(ns, fs, double(code), clickKernel)]; %#ok<AGROW>
        else
            error('Unsupported segment code %d; use 0, 2, or 7.', code);
        end
    end

    fname = sprintf('session_%02d_sequence.wav', k);
    fpath = fullfile(outDir, fname);
    audiowrite(fpath, clamp_audio(y), fs);

    totalS = numel(y) / fs;
    seqLabel = regexprep(sprintf('%d ', codes), '\s+$', '');
    manifest = [manifest; {k, string(fname), string(fpath), string(seqLabel), ...
        silenceSec, soundSec, totalS, fs, seeds(k)}]; %#ok<AGROW>
    fprintf('Wrote %s (%.2f min)\n', fpath, totalS / 60);
end

manifestPath = fullfile(outDir, 'session_sequence_manifest.csv');
writetable(manifest, manifestPath);
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

function y = clamp_audio(y)
y = max(min(y, 1.0), -1.0);
end
