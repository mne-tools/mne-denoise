function generate_aasr_filter_reference
% Generate the datafiltering2 parity fixture from the public AASR checkout.

ref_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(fileparts(ref_dir)));
aasr_repo = fullfile(repo_root, 'refs', 'asr', 'repos', 'AASR');
addpath(genpath(aasr_repo));

payload = load(fullfile(ref_dir, 'aasr_filter_input.mat'));
filtered = datafiltering2( ...
    payload.data, ...
    1:size(payload.data, 1), ...
    payload.sfreq ...
);
save(fullfile(ref_dir, 'aasr_filter_reference.mat'), '-v7', 'filtered');
end
