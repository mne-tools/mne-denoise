function run_ds004784_asr_oracle(dataset_root, oracle_root, condition, use_external, cutoff, output_path)
%RUN_DS004784_ASR_ORACLE Reproduce one released phantom ASR cell in MATLAB.

arguments
    dataset_root (1,1) string
    oracle_root (1,1) string
    condition (1,1) string
    use_external (1,1) logical
    cutoff (1,1) double {mustBePositive}
    output_path (1,1) string
end

eeglab_root = fullfile(oracle_root, "eeglab");
clean_rawdata_root = fullfile(oracle_root, "clean_rawdata");
script_root = fullfile(dataset_root, "derivatives", "Scripts");
comparison_root = fullfile(script_root, "Compare_Ground_Truth");

addpath(genpath(eeglab_root));
addpath(clean_rawdata_root);
addpath(fullfile(script_root, "Preprocessing"));
addpath(comparison_root);
addpath(fullfile(comparison_root, "helper"));

input_root = fullfile(dataset_root, "derivatives", "Data", "Imported");
target_name = "NMM10_" + condition + "_1.set";
EEG_pre = pop_loadset('filename', char(target_name), 'filepath', char(input_root));
RESULTS_pre = compareToGT(EEG_pre, 0);

params = struct("burstCrit", cutoff);
if use_external
    params.calibData = pop_loadset( ...
        'filename', 'NMM10_Clean_1.set', 'filepath', char(input_root));
end

EEG_channels = find(strcmpi('EEG', {EEG_pre.chanlocs.type}));
EEG_only = pop_select(EEG_pre, 'channel', EEG_channels);
if use_external
    calibration = pop_select(params.calibData, 'channel', {EEG_only.chanlocs.labels});
    reference_sample_mask = true(1, calibration.pnts);
else
    [~, reference_sample_mask] = clean_windows( ...
        EEG_only, 0.075, [-3.5 5.5], 1, 0.66, 0.1, 0.25);
end

tic_id = tic;
EEG_post = ASR_burst_clean(EEG_pre, params);
runtime_seconds = toc(tic_id);
RESULTS_post = compareToGT(EEG_post, 0);
correction_factor = calcCorrectionFactor(RESULTS_post, RESULTS_pre);
corrected_dqs = correction_factor * RESULTS_post.summaryMetric;

metrics = struct();
metrics.condition = char(condition);
metrics.use_external_calibration = use_external;
metrics.cutoff = cutoff;
metrics.raw_dqs = RESULTS_pre.summaryMetric;
metrics.uncorrected_dqs = RESULTS_post.summaryMetric;
metrics.correction_factor = correction_factor;
metrics.corrected_dqs = corrected_dqs;
metrics.reference_samples = nnz(reference_sample_mask);
metrics.reference_candidate_samples = numel(reference_sample_mask);
metrics.reference_fraction = mean(reference_sample_mask);
metrics.runtime_seconds = runtime_seconds;
metrics.matlab_version = version;
metrics.eeglab_commit = "5fe9e2982f350ac90d0b48c2b215ea93b63efd38";
metrics.clean_rawdata_commit = "d4b143f2a7719cf12d46c9b3e15aa827edb05614";

[output_dir, ~, ~] = fileparts(output_path);
if strlength(output_dir) > 0 && ~isfolder(output_dir)
    mkdir(output_dir);
end
cleaned_scalp_data = double(EEG_post.data(EEG_channels, :));
save(output_path, "metrics", "reference_sample_mask", "cleaned_scalp_data", "-v7.3");

json_path = replace(output_path, ".mat", ".json");
file_id = fopen(json_path, "w");
if file_id < 0
    error("Could not open JSON output %s", json_path);
end
cleanup = onCleanup(@() fclose(file_id));
fwrite(file_id, jsonencode(metrics), "char");
clear cleanup;

fprintf("corrected DQS: %.15g\n", corrected_dqs);
fprintf("reference samples: %d/%d\n", nnz(reference_sample_mask), numel(reference_sample_mask));
end
