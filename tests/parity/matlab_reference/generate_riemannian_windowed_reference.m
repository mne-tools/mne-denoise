% Direct MATLAB cross-check for ASR(method="riemannian_windowed").
%
% Loads riemannian_windowed_xcheck_input.mat (produced by
% generate_riemannian_windowed_input.py), injects the Python calibration
% state (M, T, B, A) into clean_rawdata's asr_process, and saves the
% MATLAB-cleaned output to riemannian_windowed_xcheck_reference.mat.
%
% This empirically closes the transitive chain
%   riemannian_windowed processing == standard processing  (proven in Python)
%   standard processing == MATLAB asr_process              (existing parity)
% by directly running MATLAB asr_process on the riemannian_windowed state.
%
% Run headless:
%   matlab -batch "run('tests/parity/matlab_reference/generate_riemannian_windowed_reference.m')"

here = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(fileparts(here)));
input_file = fullfile(here, 'riemannian_windowed_xcheck_input.mat');
output_file = fullfile(here, 'riemannian_windowed_xcheck_reference.mat');
clean_rawdata_path = fullfile(repo_root, 'refs', 'asr', 'repos', 'clean_rawdata');

if ~exist(input_file, 'file')
    error('Missing %s. Run generate_riemannian_windowed_input.py first.', input_file);
end
if ~exist(clean_rawdata_path, 'dir')
    error('Missing clean_rawdata path: %s', clean_rawdata_path);
end
addpath(clean_rawdata_path);
if exist('asr_process', 'file') ~= 2
    error('asr_process.m is not on the MATLAB path (%s).', clean_rawdata_path);
end

in = load(input_file);

data = in.data;
sfreq = double(in.sfreq);
window_length = double(in.window_length);
max_dims = double(in.max_dims);

% Build an asr_process state from the injected Python calibration.
state = struct();
state.M = in.M;
state.T = in.T;
state.B = in.B(:)';   % row vector
state.A = in.A(:)';   % row vector
state.cov = [];
state.carry = [];
state.iir = [];
state.last_R = [];
state.last_trivial = true;

% Mirror generate_asr_reference.m exactly: tail padding + carry trim.
stepsize = floor(sfreq * window_length / 2);
lookahead = window_length / 2;
tail_len = round(lookahead * sfreq);
tail = bsxfun(@minus, 2 * data(:, end), data(:, (end - 1):-1:(end - tail_len)));
signal_for_asr = [data tail];

maxmem = 256;
[cleaned, state_out] = asr_process( ...
    signal_for_asr, sfreq, state, window_length, lookahead, stepsize, ...
    max_dims, maxmem, false);
cleaned(:, 1:size(state_out.carry, 2)) = [];

payload = struct( ...
    'matlab_cleaned', cleaned, ...
    'sfreq', sfreq, ...
    'window_length', window_length, ...
    'max_dims', max_dims, ...
    'matlab_version', version ...
);
save(output_file, '-struct', 'payload');
fprintf('RIEMANNIAN_WINDOWED_XCHECK_OK wrote %s (cleaned %dx%d)\n', ...
    output_file, size(cleaned, 1), size(cleaned, 2));
