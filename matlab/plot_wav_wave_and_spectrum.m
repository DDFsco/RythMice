function plot_wav_wave_and_spectrum(inDir, outDir, maxWaveSec)
%PLOT_WAV_WAVE_AND_SPECTRUM Save waveform + spectrum plots for each WAV.
%   plot_wav_wave_and_spectrum('outputs/stimuli_matlab', 'outputs/figures_wav', 5)
%
% For each WAV file:
%   1) Top panel: waveform (first maxWaveSec seconds)
%   2) Bottom panel: magnitude spectrum (single-sided FFT)

if nargin < 1 || isempty(inDir), inDir = fullfile('outputs', 'stimuli_matlab'); end
if nargin < 2 || isempty(outDir), outDir = fullfile('outputs', 'figures_wav'); end
if nargin < 3 || isempty(maxWaveSec), maxWaveSec = 5; end

if ~exist(inDir, 'dir')
    error('Input directory does not exist: %s', inDir);
end
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

files = dir(fullfile(inDir, '*.wav'));
if isempty(files)
    error('No WAV files found in: %s', inDir);
end

for i = 1:numel(files)
    wavPath = fullfile(files(i).folder, files(i).name);
    [y, fs] = audioread(wavPath);
    if size(y, 2) > 1
        y = mean(y, 2);
    end

    n = numel(y);
    t = (0:n-1)' / fs;

    nWave = min(n, max(1, round(maxWaveSec * fs)));
    tWave = t(1:nWave);
    yWave = y(1:nWave);

    NFFT = 2^nextpow2(n);
    Y = fft(y, NFFT);
    P2 = abs(Y / n);
    P1 = P2(1:NFFT/2+1);
    if numel(P1) > 2
        P1(2:end-1) = 2 * P1(2:end-1);
    end
    f = fs * (0:(NFFT/2)) / NFFT;

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 700]);

    subplot(2,1,1);
    plot(tWave, yWave, 'k-', 'LineWidth', 0.8);
    grid on;
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(sprintf('Waveform (first %.2f s): %s', tWave(end), files(i).name), 'Interpreter', 'none');

    subplot(2,1,2);
    plot(f, P1, 'b-', 'LineWidth', 1.0);
    xlim([0 min(2000, fs/2)]);
    grid on;
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    title('Single-Sided Magnitude Spectrum');

    [~, baseName, ~] = fileparts(files(i).name);
    outPng = fullfile(outDir, sprintf('%s_wave_spectrum.png', baseName));
    exportgraphics(fig, outPng, 'Resolution', 160);
    close(fig);

    fprintf('Saved: %s\n', outPng);
end

fprintf('Done. Plots saved to: %s\n', outDir);
end
