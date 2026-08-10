% Generate MATLAB clean_rawdata ASR reference outputs for mne-denoise.
%
% Prerequisites:
%   1. Run generate_asr_input.py from the repository root.
%   2. Add clean_rawdata to the MATLAB path so clean_windows.m,
%      asr_calibrate.m, and asr_process.m are visible.
%
% Outputs:
%   - asr_reference_results.mat (legacy first-case artifact)
%   - asr_case_reference_<name>.mat (expanded case matrix)

here = fileparts(mfilename('fullpath'));
legacy_input_file = fullfile(here, 'asr_input_fixture.mat');
legacy_output_file = fullfile(here, 'asr_reference_results.mat');
case_filter = getenv('MNE_DENOISE_ASR_CASE');

if isempty(case_filter) && ~exist(legacy_input_file, 'file')
    error('Missing %s. Run generate_asr_input.py first.', legacy_input_file);
end
if exist('clean_windows', 'file') ~= 2
    error('clean_windows.m is not on the MATLAB path.');
end
if exist('asr_calibrate', 'file') ~= 2
    error('asr_calibrate.m is not on the MATLAB path.');
end
if exist('asr_process', 'file') ~= 2
    error('asr_process.m is not on the MATLAB path.');
end

inputs = dir(fullfile(here, 'asr_case_input_*.mat'));
if isempty(inputs)
    error('No asr_case_input_*.mat files found. Run generate_asr_input.py first.');
end
[~, order] = sort({inputs.name});
inputs = inputs(order);
if ~isempty(case_filter)
    inputs = inputs(contains({inputs.name}, case_filter));
    if isempty(inputs)
        error('No ASR input fixture matched MNE_DENOISE_ASR_CASE=%s.', case_filter);
    end
end

for idx = 1:numel(inputs)
    input_path = fullfile(here, inputs(idx).name);
    case_name = erase(inputs(idx).name, 'asr_case_input_');
    case_name = erase(case_name, '.mat');
    output_path = fullfile(here, sprintf('asr_case_reference_%s.mat', case_name));

    S = load(input_path);
    [payload, legacy_payload] = compute_case_reference(S);

    save(output_path, '-struct', 'payload', '-v7');
    fprintf('Wrote %s\n', output_path);

    if idx == 1 && isempty(case_filter)
        save(legacy_output_file, '-struct', 'legacy_payload', '-v7');
        fprintf('Wrote %s\n', legacy_output_file);
    end
end


function [payload, legacy_payload] = compute_case_reference(S)
data = double(S.data);
calibration = double(S.calibration);
sfreq = double(S.sfreq);
cutoff = double(S.cutoff);
blocksize = double(S.blocksize);
window_length = double(S.window_length);
window_overlap = double(S.window_overlap);
max_dropout_fraction = double(S.max_dropout_fraction);
min_clean_fraction = double(S.min_clean_fraction);
max_dims = double(S.max_dims);
maxmem = double(S.maxmem);
B = double(S.filter_b);
A = double(S.filter_a);
use_auto_calibration = logical(S.use_auto_calibration);
ref_max_bad_channels = double(S.ref_max_bad_channels);
ref_tolerances = double(S.ref_tolerances(:))';
ref_window_length = double(S.ref_window_length);

if use_auto_calibration
    signal = make_signal_struct(data, sfreq);
    [ref_section, reference_sample_mask] = clean_windows( ...
        signal, ref_max_bad_channels, ref_tolerances, ref_window_length, ...
        window_overlap, max_dropout_fraction, min_clean_fraction);
    calibration_used = double(ref_section.data);
else
    calibration_used = calibration;
    reference_sample_mask = false(1, 0);
end

state = asr_calibrate( ...
    calibration_used, sfreq, cutoff, blocksize, B, A, window_length, ...
    window_overlap, max_dropout_fraction, min_clean_fraction, maxmem);

stepsize = floor(sfreq * window_length / 2);
lookahead = window_length / 2;
tail_len = round(lookahead * sfreq);
tail = bsxfun(@minus, 2 * data(:, end), data(:, (end - 1):-1:(end - tail_len)));
signal_for_asr = [data tail];

[cleaned, state_out] = asr_process( ...
    signal_for_asr, sfreq, state, window_length, lookahead, stepsize, ...
    max_dims, maxmem, false);
cleaned(:, 1:size(state_out.carry, 2)) = [];

M = state.M;
T = state.T;
thresholds = sqrt(sum(T.^2, 2));
B = state.B;
A = state.A;

payload = struct( ...
    'data', data, ...
    'calibration', calibration, ...
    'calibration_used', calibration_used, ...
    'reference_sample_mask', logical(reference_sample_mask), ...
    'use_auto_calibration', use_auto_calibration, ...
    'ref_max_bad_channels', ref_max_bad_channels, ...
    'ref_tolerances', ref_tolerances, ...
    'ref_window_length', ref_window_length, ...
    'sfreq', sfreq, ...
    'cutoff', cutoff, ...
    'blocksize', blocksize, ...
    'window_length', window_length, ...
    'window_overlap', window_overlap, ...
    'max_dropout_fraction', max_dropout_fraction, ...
    'min_clean_fraction', min_clean_fraction, ...
    'max_dims', max_dims, ...
    'maxmem', maxmem, ...
    'B', B, ...
    'A', A, ...
    'M', M, ...
    'T', T, ...
    'thresholds', thresholds, ...
    'cleaned', cleaned ...
);

legacy_payload = rmfield(payload, {'calibration_used', 'reference_sample_mask', ...
    'use_auto_calibration', 'ref_max_bad_channels', 'ref_tolerances', ...
    'ref_window_length'});
end


function signal = make_signal_struct(data, sfreq)
signal = struct();
signal.data = data;
signal.srate = sfreq;
signal.nbchan = size(data, 1);
signal.pnts = size(data, 2);
signal.trials = 1;
signal.xmin = 0;
signal.xmax = (signal.pnts - 1) / signal.srate;
signal.event = [];
signal.urevent = [];
signal.epoch = [];
signal.icaact = [];
signal.reject = [];
signal.stats = [];
signal.specdata = [];
signal.specicaact = [];
signal.etc = struct();
end
