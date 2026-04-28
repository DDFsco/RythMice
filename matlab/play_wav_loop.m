function play_wav_loop(wavPath, nLoops, pauseBetweenLoopsSec)
%PLAY_WAV_LOOP Play one WAV repeatedly in MATLAB.
%   play_wav_loop('outputs/stimuli_matlab/periodic_8hz.wav', 10, 0)
%
% Press Ctrl+C in command window to stop early.

if nargin < 2 || isempty(nLoops), nLoops = Inf; end
if nargin < 3 || isempty(pauseBetweenLoopsSec), pauseBetweenLoopsSec = 0; end

[y, fs] = audioread(wavPath);

loopCounter = 0;
while loopCounter < nLoops
    player = audioplayer(y, fs);
    playblocking(player);
    loopCounter = loopCounter + 1;

    if pauseBetweenLoopsSec > 0
        pause(pauseBetweenLoopsSec);
    end
end
end
