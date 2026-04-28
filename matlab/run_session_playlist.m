function schedule = run_session_playlist(stimDir, freqsHz, includeRandom, onSec, offSec, baselineSec, postSec, seed)
%RUN_SESSION_PLAYLIST Run randomized stimulus ON/OFF session from MATLAB.
%   This script plays:
%   baseline silence -> [stim ON + silence OFF]*N -> post silence
%
% Example:
%   run_session_playlist('outputs/stimuli_matlab', [2 6 8 12], true, 120, 300, 1200, 600, 7);

if nargin < 1 || isempty(stimDir), stimDir = fullfile('outputs', 'stimuli_matlab'); end
if nargin < 2 || isempty(freqsHz), freqsHz = [2 6 8 12]; end
if nargin < 3 || isempty(includeRandom), includeRandom = true; end
if nargin < 4 || isempty(onSec), onSec = 120; end
if nargin < 5 || isempty(offSec), offSec = 300; end
if nargin < 6 || isempty(baselineSec), baselineSec = 1200; end
if nargin < 7 || isempty(postSec), postSec = 600; end
if nargin < 8 || isempty(seed), seed = 7; end

rng(seed);

conds = {};
for i = 1:numel(freqsHz)
    hz = freqsHz(i);
    conds{end+1,1} = sprintf('periodic_%dhz.wav', round(hz)); %#ok<AGROW>
    conds{end,2} = hz;
    conds{end,3} = 0;
    if includeRandom
        conds{end+1,1} = sprintf('random_%dhz.wav', round(hz)); %#ok<AGROW>
        conds{end,2} = hz;
        conds{end,3} = 1;
    end
end

order = randperm(size(conds,1));
conds = conds(order, :);

schedule = table('Size', [0 6], ...
    'VariableTypes', {'double','double','string','string','double','double'}, ...
    'VariableNames', {'start_s','end_s','block_type','stimulus_name','frequency_hz','is_random'});

fprintf('Starting baseline silence: %.1f s\n', baselineSec);
t0 = tic;
pause(baselineSec);
schedule = [schedule; {0, baselineSec, "baseline_silence", "silence.wav", NaN, NaN}];

for i = 1:size(conds, 1)
    stimName = conds{i,1};
    hz = conds{i,2};
    isRand = conds{i,3};
    stimPath = fullfile(stimDir, stimName);

    if ~isfile(stimPath)
        error('Missing stimulus file: %s', stimPath);
    end

    startOn = toc(t0);
    fprintf('ON  - %s (%.1f s)\n', stimName, onSec);
    play_for_duration(stimPath, onSec);
    endOn = toc(t0);
    schedule = [schedule; {startOn, endOn, "stim_on", string(stimName), hz, isRand}];

    startOff = toc(t0);
    fprintf('OFF - silence (%.1f s)\n', offSec);
    pause(offSec);
    endOff = toc(t0);
    schedule = [schedule; {startOff, endOff, "matched_silence", "silence.wav", hz, isRand}];
end

startPost = toc(t0);
fprintf('Post-session silence: %.1f s\n', postSec);
pause(postSec);
endPost = toc(t0);
schedule = [schedule; {startPost, endPost, "post_silence", "silence.wav", NaN, NaN}];

outDir = fullfile('outputs', 'session_plans_matlab');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
outCsv = fullfile(outDir, sprintf('session_%s.csv', datestr(now, 'yyyymmdd_HHMMSS')));
writetable(schedule, outCsv);
fprintf('Saved schedule log: %s\n', outCsv);
end

function play_for_duration(wavPath, durationSec)
[y, fs] = audioread(wavPath);
player = audioplayer(y, fs);
startTime = tic;
while toc(startTime) < durationSec
    playblocking(player);
end
end
