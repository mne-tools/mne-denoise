function generate_aasr_reference
% Generate MATLAB parity fixtures for AdaptiveASR against the local AASR repo.

ref_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(fileparts(ref_dir)));
aasr_repo = fullfile(repo_root, 'refs', 'asr', 'repos', 'AASR');
addpath(genpath(aasr_repo));

inputs = dir(fullfile(ref_dir, 'aasr_case_input_*.mat'));
variants = {'psp', 'psw'};
update_counts = [1 3];

for input_idx = 1:numel(inputs)
    payload = load(fullfile(inputs(input_idx).folder, inputs(input_idx).name));
    case_name = char(payload.case_name);

    for variant_idx = 1:numel(variants)
        variant = variants{variant_idx};
        for count_idx = 1:numel(update_counts)
            n_updates = update_counts(count_idx);
            if strcmp(variant, 'psp')
                obj = ASR(payload.cutoff, payload.sfreq);
            else
                obj = ASR_PSW(payload.cutoff, payload.sfreq);
            end

            for update_idx = 1:n_updates
                chunk_field = sprintf('update_chunk_%d', update_idx);
                chunk = payload.(chunk_field);
                obj = obj.update(chunk);
            end

            cleaned = obj.reconstruct(payload.data);
            patterns = obj.fsm.get_components([]);
            thresholds = sqrt(sum(obj.state.T.^2, 2));

            out_path = fullfile( ...
                ref_dir, ...
                sprintf('aasr_case_reference_%s_%s_%dupdates.mat', case_name, variant, n_updates) ...
            );
            data = payload.data; %#ok<NASGU>
            sfreq = payload.sfreq; %#ok<NASGU>
            cutoff = payload.cutoff; %#ok<NASGU>
            window_length = payload.window_length; %#ok<NASGU>
            clean_window_length = payload.clean_window_length; %#ok<NASGU>
            clean_window_overlap = payload.clean_window_overlap; %#ok<NASGU>
            clean_max_bad_channels = payload.clean_max_bad_channels; %#ok<NASGU>
            clean_tolerances = payload.clean_tolerances; %#ok<NASGU>
            blocksize = payload.blocksize; %#ok<NASGU>
            max_dims = payload.max_dims; %#ok<NASGU>
            learning_rate = payload.learning_rate; %#ok<NASGU>
            tau_field = sprintf('tau_%s', variant);
            tau = payload.(tau_field); %#ok<NASGU>
            update_chunk_1 = payload.update_chunk_1; %#ok<NASGU>
            update_chunk_2 = payload.update_chunk_2; %#ok<NASGU>
            update_chunk_3 = payload.update_chunk_3; %#ok<NASGU>
            M = obj.state.M; %#ok<NASGU>
            T = obj.state.T; %#ok<NASGU>
            save( ...
                out_path, ...
                'case_name', 'variant', 'n_updates', ...
                'data', 'sfreq', 'cutoff', ...
                'window_length', 'clean_window_length', 'clean_window_overlap', ...
                'clean_max_bad_channels', 'clean_tolerances', ...
                'blocksize', 'max_dims', 'learning_rate', 'tau', ...
                'update_chunk_1', 'update_chunk_2', 'update_chunk_3', ...
                'M', 'T', 'thresholds', 'patterns', 'cleaned' ...
            );
        end
    end
end
