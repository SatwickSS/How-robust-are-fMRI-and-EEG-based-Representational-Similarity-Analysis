function perform_sca(subjects)
    % USAGE:
    %   perform_sca([1,2,3,4,5])
    %
    % INPUT:
    %   subjects: A row vector of subject IDs to be included in the analysis.
    %             Example: [1,2,3,4,5]

    %% --- Configuration ---
    fprintf('--- CONFIGURING ANALYSIS PARAMETERS ---\n');
    
    % Add CoSMoMVPA to path using relative path
    addpath('../../tools/CoSMoMVPA/mvpa');
    
    % --- Key Analysis Parameters ---
    lobes = {'whole_brain', 'frontal', 'central', 'temporal', 'parietal_occipital'};
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    nShuf = 10;
    time_specs = {150, 200, 250, 300, 'average'};
    avg_window_ms = [0, 996];
    
    % --- Path Configuration (using hard-coded relative paths) ---
    base_path_nck = '../../derivatives/rdms/';
    base_path_k_shuf = '../../derivatives/rdms_shuffle/';
    base_path_lobe_ori = '../../derivatives/lobes/';
    base_path_lobe_shuf = '../../derivatives/lobes_shuffle/';
    save_path = '../../results/sca_single_reference_results/';
    
    fprintf('  Subjects to process: %s\n', mat2str(subjects));
    fprintf('  Results will be saved to: %s\n', save_path);
    if ~exist(save_path, 'dir'), fprintf('  Save directory not found. Creating it...\n'); mkdir(save_path); end
    
    %% PHASE 1: Load the SINGLE Reference RDM (Whole Brain, k=1, 200ms)
    fprintf('\n--- PHASE 1: Loading the single reference RDM for all subjects ---\n');
    ref_rdm = struct();
    for sub = subjects
        fprintf('  Loading reference for Subject %02i...\n', sub);
        fn_ref = fullfile(base_path_nck, sprintf('sub%02i/sub-%02i_rdm_test_images_k01.mat', sub, sub));
        data_ref = load(fn_ref, 'res');
        time_axis = data_ref.res.a.fdim.values{1};
        ref_rdm.(sprintf('sub%02i', sub)) = extract_rdm_at_time(data_ref, 200, time_axis, avg_window_ms);
    end
    fprintf('--- Reference RDM loaded successfully. ---\n');
    
    %% PHASE 2: Load ALL Specification RDMs (Real and Shuffled)
    fprintf('\n--- PHASE 2: Loading all specification RDMs (this may take time) ---\n');
    all_spec_rdms = struct();
    all_shuf_rdms = struct();
    
    for t_idx = 1:numel(time_specs)
        current_time_spec = time_specs{t_idx};
        if isnumeric(current_time_spec)
            time_field = sprintf('t%dms', current_time_spec);
        else
            time_field = 't_avg';
        end
        
        fprintf('  Processing Time Spec: %s\n', time_field);
        for l_idx = 1:numel(lobes)
            current_lobe = lobes{l_idx};
            fprintf('    Processing Lobe: %s\n', current_lobe);
            for k_idx = 1:numel(k_values)
                current_k = k_values(k_idx);
                
                if strcmp(current_lobe, 'whole_brain') && current_k == 1 && isnumeric(current_time_spec) && current_time_spec == 200
                    fprintf('      Skipping whole_brain, k=1, 200ms, as it is the reference specification.\n');
                    continue;
                end
                
                spec_name = sprintf('%s_k%02d_%s', current_lobe, current_k, time_field);
                fprintf('      Processing Specification: %s\n', spec_name);
                
                for sub = subjects
                    subj_field = sprintf('sub%02i', sub);
                    
                    % Load real spec RDM
                    if strcmp(current_lobe, 'whole_brain')
                        fn_spec = fullfile(base_path_nck, sprintf('sub%02i/sub-%02i_rdm_test_images_k%02i.mat', sub, sub, current_k));
                        data_spec = load(fn_spec, 'res');
                        time_axis = data_spec.res.a.fdim.values{1};
                        all_spec_rdms.(subj_field).(spec_name) = extract_rdm_at_time(data_spec, current_time_spec, time_axis, avg_window_ms);
                    else
                        fn_spec = fullfile(base_path_lobe_ori, sprintf('sub%02i/sub-%02i_rdm_test_images_lobes_k%02i.mat', sub, sub, current_k));
                        data_spec = load(fn_spec, 'res');
                        temp_struct = struct(); temp_struct.res = data_spec.res.(current_lobe);
                        time_axis = temp_struct.res.a.fdim.values{1};
                        all_spec_rdms.(subj_field).(spec_name) = extract_rdm_at_time(temp_struct, current_time_spec, time_axis, avg_window_ms);
                    end
                    
                    % Load shuffled spec RDMs
                    all_shuf_rdms.(subj_field).(spec_name) = cell(nShuf, 1);
                    for shuf = 1:nShuf
                        if strcmp(current_lobe, 'whole_brain')
                            fn_shuf = fullfile(base_path_k_shuf, sprintf('shuf_%02i/k_%02i/sub-%02i_rdm_test_images_k%02i.mat', shuf, current_k, sub, current_k));
                            data_shuf = load(fn_shuf, 'res');
                            time_axis = data_shuf.res.a.fdim.values{1};
                            all_shuf_rdms.(subj_field).(spec_name){shuf} = extract_rdm_at_time(data_shuf, current_time_spec, time_axis, avg_window_ms);
                        else
                            fn_shuf = fullfile(base_path_lobe_shuf, sprintf('sub%02i/sub-%02i_rdm_test_images_lobes_s%02i_k%02i.mat', sub, sub, shuf, current_k));
                            data_shuf = load(fn_shuf, 'res');
                            temp_struct_shuf = struct(); temp_struct_shuf.res = data_shuf.res.(current_lobe);
                            time_axis = temp_struct_shuf.res.a.fdim.values{1};
                            all_shuf_rdms.(subj_field).(spec_name){shuf} = extract_rdm_at_time(temp_struct_shuf, current_time_spec, time_axis, avg_window_ms);
                        end
                    end
                end
            end
        end
    end
    fprintf('--- All specification RDMs loaded successfully. ---\n');
    
    
    %% PHASE 3: Calculate All Correlations (Real and Shuffled)
    fprintf('\n--- PHASE 3: Calculating all subject-level correlations ---\n');
    all_tau_values = struct();
    all_shuf_tau_distributions = struct();
    spec_names = fieldnames(all_spec_rdms.(sprintf('sub%02i', subjects(1))));
    
    for sub = subjects
        subj_field = sprintf('sub%02i', sub);
        fprintf('  Processing correlations for Subject %s\n', subj_field);
        for i_spec = 1:numel(spec_names)
            spec_name = spec_names{i_spec};
            
            all_tau_values.(subj_field).(spec_name) = corr(ref_rdm.(subj_field), all_spec_rdms.(subj_field).(spec_name), 'type', 'Kendall');
            
            shuf_taus = zeros(nShuf, 1);
            for shuf = 1:nShuf
                shuf_taus(shuf) = corr(ref_rdm.(subj_field), all_shuf_rdms.(subj_field).(spec_name){shuf}, 'type', 'Kendall');
            end
            all_shuf_tau_distributions.(subj_field).(spec_name) = shuf_taus;
        end
    end
    fprintf('--- All correlation calculations complete. ---\n');
    
    %% PHASE 4: Calculate Subject-Level P-values from Correlations (Permutation Test)
    fprintf('\n--- PHASE 4: Performing permutation test to calculate subject-level p-values ---\n');
    all_subject_p_values = struct();
    for sub = subjects
        subj_field = sprintf('sub%02i', sub);
        fprintf('  Calculating p-values for Subject %s\n', subj_field);
        for i_spec = 1:numel(spec_names)
            spec_name = spec_names{i_spec};
            
            real_tau = all_tau_values.(subj_field).(spec_name);
            null_distribution = all_shuf_tau_distributions.(subj_field).(spec_name);
            
            nShuffles = numel(null_distribution);
            percentile = sum(null_distribution < real_tau) / nShuffles * 100;
            p_val = 2 * min((1 - percentile/100), (percentile/100));
            
            all_subject_p_values.(subj_field).(spec_name) = p_val;
        end
    end
    fprintf('--- All subject-level p-value calculations complete. ---\n');
    
    %% PHASE 5: Group-level aggregation, Plotting, and Saving
    fprintf('\n--- PHASE 5: Aggregating to group-level, plotting, and saving all results ---\n');
    group_p_fisher = struct();
    group_p_stouffer = struct();
    
    fprintf('  Step 5.1: Aggregating p-values to group level...\n');
    for i_spec = 1:numel(spec_names)
        spec_name = spec_names{i_spec};
        
        p_vals_across_subs = [];
        for sub = subjects
            p_vals_across_subs = [p_vals_across_subs; all_subject_p_values.(sprintf('sub%02i', sub)).(spec_name)];
        end
        
        p_vals_across_subs(p_vals_across_subs == 0) = eps;
        p_vals_across_subs(p_vals_across_subs == 1) = 1 - eps;
        
        chi2_stat = -2 * sum(log(p_vals_across_subs)); df = 2 * length(p_vals_across_subs);
        group_p_fisher.(spec_name) = 1 - chi2cdf(chi2_stat, df);
        
        z_scores = norminv(1 - p_vals_across_subs); z_combined = sum(z_scores) / sqrt(length(z_scores));
        group_p_stouffer.(spec_name) = 1 - normcdf(z_combined);
    end
    fprintf('  Group-level aggregation complete.\n');
    
    % Sort and Plot (using Stouffer's as the primary method for sorting)
    fprintf('  Step 5.2: Generating final plot with specified style...\n');
    fig = figure('Color', 'white', 'Visible', 'off');
    p_values_for_plot = cell2mat(struct2cell(group_p_stouffer));
    [sorted_p_values, ~] = sort(p_values_for_plot);
    % Plot the p-values line (solid, thick blue)
    
    plot(sorted_p_values, 'LineWidth', 4, 'Color', [0.12, 0.47, 0.71]); % A standard blue color
    hold on;
    
    % Plot the 0.025 significance level line (dashed, thick magenta)
    plot([0, numel(sorted_p_values)+1], [0.025, 0.025], '--', 'Color', [1, 0, 1], 'LineWidth', 4);
    
    plot(1, sorted_p_values(1), 'o', 'MarkerSize', 15, 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'red');
    
    % --- Axis and Figure Styling to match the target image ---
    ax = gca;
    box off; % Removes the top and right axis lines
    ax.LineWidth = 3;       % Thick axes lines
    ax.TickDir = 'out';     % Ticks point outwards
    ax.FontSize = 24;       % Larger font for tick labels
    ax.FontName = 'Arial';  % Clear, sans-serif font
    grid off;               % Ensure no grid is visible
    
    % Set axis limits and ticks to match the reference image
    xlim([-5, 255]); % Give a little space on the left for the marker
    ylim([-0.002, 0.03]); % Y-axis from slightly below 0 to 0.03
    ax.YTick = [0.00, 0.01, 0.02];
    ax.YTickLabel = {'0.00', '0.01', '0.02'}; % Ensure two decimal places
    
    % Label the axes with matching style
    xlabel('Specifications(sorted by p-values)', 'FontSize', 28, 'FontName', 'Arial');
    ylabel('p-values', 'FontSize', 28, 'FontName', 'Arial');
    
    % Add legend matching the target image
    lgd=legend({'p-values', '0.025 significance level', 'Original Specification'}, 'Location', 'northwest', 'FontSize', 9);
    lgd.Position=[0.20, 0.55, 0.35, 0.15]; % Adjust position to match the target image
    hold off;
    
    % Save the final styled figure
    plot_filename = fullfile(save_path, 'sca_plot_single_reference.png');
    print(fig, plot_filename, '-dpng', '-r300');
    close(fig);
    fprintf('    -> plot saved successfully.\n');
    
    
    
    % Save all data structures
    fprintf('  Step 5.3: Saving all generated data structures to a single .mat file...\n');
    data_filename = fullfile(save_path, 'sca_full_analysis_data.mat');
    save(data_filename, ...
        'ref_rdm', ...
        'all_spec_rdms', ...
        'all_shuf_rdms', ...
        'all_tau_values', ...
        'all_shuf_tau_distributions', ...
        'all_subject_p_values', ...
        'group_p_stouffer', ...
        'group_p_fisher', ...
        '-v7.3');
    fprintf('    -> Full data file saved successfully to: %s\n', data_filename);
    
    fprintf('\n\n--- ANALYSIS COMPLETE ---\n');
    
    end 
    
    function rdm_vector = extract_rdm_at_time(data_struct, time_spec, time_axis_ms, avg_window_ms)
        time_axis_sec = time_axis_ms / 1000;
        if isnumeric(time_spec)
            time_in_sec = time_spec / 1000;
            [~, time_idx] = min(abs(time_axis_sec - time_in_sec));
            rdm_vector = data_struct.res.samples(:, time_idx);
        elseif ischar(time_spec) && strcmpi(time_spec, 'average')
            [~, start_idx] = min(abs(time_axis_sec - (avg_window_ms(1) / 1000)));
            [~, end_idx] = min(abs(time_axis_sec - (avg_window_ms(2) / 1000)));
            rdm_vector = mean(data_struct.res.samples(:, start_idx:end_idx), 2);
        else
            error('Invalid time_spec provided.');
        end
    end