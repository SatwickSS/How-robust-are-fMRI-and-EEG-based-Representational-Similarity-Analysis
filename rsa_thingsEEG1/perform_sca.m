
function perform_sca(subjects)
    % Takes input a list of subjects. 
    % eg: perform_sca([1,2,3,4,5])

    addpath('../../CoSMoMVPA/mvpa'); 

    %% Define parameters 
    lobes = {'frontal', 'central', 'temporal', 'parietal_occipital'};  % Brain regions
    k_values = [1:10];  % k-values
    nSubs = numel(subjects);  % Number of subjects
    nK = numel(k_values);  % Number of k-values
    nShuf = 10;  % Number of shuffle iterations

    %% initialize storage structures

    rdm_ori = struct();
    rdm_k = struct();
    rdm_k_shuf = struct();
    rdm_spec = struct();
    rdm_shuf = struct();
    %% Load original and spec data for each subject at 200 ms

    for sub = subjects
        fprintf('Processing subject %02i...\n', sub);
        
        % Load original data for subject
        fn_ori = sprintf('../../derivatives/rdms/sub%02i/sub-%02i_rdm_test_images_k01.mat', sub, sub);
        fprintf('Loading original data: %s\n', fn_ori);
        data_ori = load(fn_ori, 'res');
        % rdm_ori.(sprintf('sub%02i', sub)) = max(data_ori.res.samples, [], 2);
        rdm_ori.(sprintf('sub%02i', sub)) = data_ori.res.samples(:,200); 
        
        % Load spec data for subject
        for k = k_values

            kval = sprintf('k%02i', k);

            fn_k = sprintf('../../derivatives/rdms/sub%02i/sub-%02i_rdm_test_images_k%02i.mat', sub, sub, k); 
            fprintf('Loading spec(k) data: %s\n', fn_k);
            data_k = load(fn_k, 'res');
            % rdm_k.(sprintf('sub%02i', sub)).(kval) = max(data_k.res.samples, [], 2);
            rdm_k.(sprintf('sub%02i', sub)).(kval) = data_k.res.samples(:,200); %both 200ms and max give similar results
    

            fn_spec = sprintf('../../derivatives/lobes/sub%02i/sub-%02i_rdm_test_images_lobes_k%02i.mat', sub, sub, k);
            fprintf('Loading spec data: %s\n', fn_spec);
            data_spec = load(fn_spec, 'res');
            for i = 1:4  % Loop over lobes
                lobename = lobes{i};
                % rdm_spec.(sprintf('sub%02i', sub)).(kval).(lobename) = max(data_spec.res.(lobename).samples, [], 2);
                rdm_spec.(sprintf('sub%02i', sub)).(kval).(lobename) = data_spec.res.(lobename).samples(:,200); %both 200ms and max give similar results
            end
            
            % Load shuffled data for subject, k, and shuffle (shuf=1 to shuf=5)
            for shuf = 1:nShuf

                fn_k_shuf = sprintf('../../derivatives/rdms_shuffle/shuf_%02i/k_%02i/sub-%02i_rdm_test_images_k%02i.mat', shuf, k, sub, k);
                fprintf('Loading shuffled(k) data: %s\n', fn_k_shuf);
                data_k_shuf = load(fn_k_shuf, 'res');
                shufnamek = sprintf('shuf_%02i', shuf);
                % rdm_k_shuf.(sprintf('sub%02i', sub)).(shufnamek).(kval) = max(data_k_shuf.res.samples, [], 2);
                rdm_k_shuf.(sprintf('sub%02i', sub)).(shufnamek).(kval) = data_k_shuf.res.samples(:,200);


                fn_shuf = sprintf('.../../derivatives/lobes_shuffle/sub%02i/sub-%02i_rdm_test_images_lobes_s%02i_k%02i.mat', sub, sub, shuf, k);
                fprintf('Loading shuffled data: %s\n', fn_shuf);
                data_shuf = load(fn_shuf, 'res');
                shufname = sprintf('shuf_%02i', shuf);
                for i = 1:4  % Loop over lobes
                    lobename = lobes{i};
                    % rdm_shuf.(sprintf('sub%02i', sub)).(shufname).(kval).(lobename) = max(data_shuf.res.(lobename).samples, [], 2);
                    rdm_shuf.(sprintf('sub%02i', sub)).(shufname).(kval).(lobename) = data_shuf.res.(lobename).samples(:,200);
                end
            end


        end
    end
    %%

    save('../../results/rdm_ori_new.mat', 'rdm_ori');
    save('../../results/rdm_k_new.mat', 'rdm_k');
    save('../../results/rdm_k_shuf_new.mat', 'rdm_k_shuf');
    save('../../results/rdm_spec_new.mat', 'rdm_spec');
    save('../../results/rdm_shuf_new.mat', 'rdm_shuf');


    %% Calculate Kendall's Tau correlation between original and spec data


    tau_ori_k = struct();
    p_values_ori_k = struct();

    tau_ori_shuf_k = struct();
    p_values_ori_shuf_k = struct();

    tau_ori_spec = struct();
    p_values_ori_spec = struct();

    tau_ori_shuf = struct();
    p_values_ori_shuf = struct();

    for sub = subjects
        fprintf('Processing subject %02i...\n', sub);
        
        for k = k_values
            kval = sprintf('k%02i', k);

            [tau_ori_k.(sprintf('sub%02i', sub)).(kval), p_values_ori_k.(sprintf('sub%02i', sub)).(kval)] = corr(rdm_ori.(sprintf('sub%02i', sub))(:), rdm_k.(sprintf('sub%02i', sub)).(kval)(:), 'type', 'Kendall');


            for i = 1:4
                lobename = lobes{i};
                [tau_ori_spec.(sprintf('sub%02i', sub)).(sprintf('%s_%s', kval, lobename)), p_values_ori_spec.(sprintf('sub%02i', sub)).(sprintf('%s_%s', kval, lobename))] = corr(rdm_ori.(sprintf('sub%02i', sub))(:), rdm_spec.(sprintf('sub%02i', sub)).(kval).(lobename)(:), 'type', 'Kendall');
            end

            for shuf = 1:nShuf
                shufname = sprintf('shuf_%02i', shuf);
                [tau_ori_shuf_k.(sprintf('sub%02i', sub)).(shufname).(kval), p_values_ori_shuf_k.(sprintf('sub%02i', sub)).(shufname).(kval)] = corr(rdm_ori.(sprintf('sub%02i', sub))(:), rdm_k_shuf.(sprintf('sub%02i', sub)).(shufname).(kval)(:), 'type', 'Kendall');
                for i = 1:4
                    lobename = lobes{i};
                    [tau_ori_shuf.(sprintf('sub%02i', sub)).(shufname).(sprintf('%s_%s', kval, lobename)), p_values_ori_shuf.(sprintf('sub%02i', sub)).(shufname).(sprintf('%s_%s', kval, lobename))] = corr(rdm_ori.(sprintf('sub%02i', sub))(:), rdm_shuf.(sprintf('sub%02i', sub)).(shufname).(kval).(lobename)(:), 'type', 'Kendall');
                end
            end

        end
    end

    save('../../results/tau_ori_k_new.mat', 'tau_ori_k');
    save('../../results/p_values_ori_k_new.mat', 'p_values_ori_k');

    save('../../results/tau_ori_shuf_k_new.mat', 'tau_ori_shuf_k');
    save('../../results/p_values_ori_shuf_k_new.mat', 'p_values_ori_shuf_k');

    save('../../results/tau_ori_spec_new.mat', 'tau_ori_spec');
    save('../../results/p_values_ori_spec_new.mat', 'p_values_ori_spec');

    save('../../results/tau_ori_shuf_new.mat', 'tau_ori_shuf');
    save('../../results/p_values_ori_shuf_new.mat', 'p_values_ori_shuf');



    %% Calculate percentile of tau values

    % Initialize structure to store percentiles
    percentile_ori_vs_shuf = struct();
    percentile_k_vs_kshuf = struct();

    for sub = subjects
        subj_field = sprintf('sub%02i', sub);
        for k = k_values
            kval = sprintf('k%02i', k);

            % Collect tau values from all shuffle iterations for rdm_k and rdm_k_shuf

            tau_ori_kvalue = tau_ori_k.(subj_field).(kval);
            tau_shuf_kvalue = zeros(nShuf, 1);

            for shuf = 1:nShuf
                shufname = sprintf('shuf_%02i', shuf);
                tau_shuf_kvalue(shuf) = tau_ori_shuf_k.(subj_field).(shufname).(kval);
            end

            % Calculate the percentile of tau_ori relative to the shuffled taus.    
            percentile = sum(tau_shuf_kvalue < tau_ori_kvalue) / nShuf * 100;

            % Store the result

            percentile_k_vs_kshuf.(subj_field).(kval) = percentile;

            for i = 1:length(lobes)
                lobename = lobes{i};
                
                % Get the original tau for this specification
                tau_ori = tau_ori_spec.(subj_field).(sprintf('%s_%s', kval, lobename));
                
                % Collect tau values from all shuffle iterations for this subject/specification
                tau_shuf_vals = zeros(nShuf, 1);
                for shuf = 1:nShuf
                    shufname = sprintf('shuf_%02i', shuf);
                    tau_shuf_vals(shuf) = tau_ori_shuf.(subj_field).(shufname).(sprintf('%s_%s', kval, lobename));
                end
                
                % Calculate the percentile of tau_ori relative to the shuffled taus.
                % This is the fraction of shuffled taus that are less than tau_ori, multiplied by 100.
                percentile = sum(tau_shuf_vals < tau_ori) / nShuf * 100;
                
                % Store the result
                percentile_ori_vs_shuf.(subj_field).(sprintf('%s_%s', kval, lobename)) = percentile;
            end
        end
    end


    save('../../results/percentile_k_vs_kshuf_new.mat', 'percentile_k_vs_kshuf');
    save('../../results/percentile_ori_vs_shuf_new.mat', 'percentile_ori_vs_shuf');

    %% Calculate p-values from percentiles

    % Initialize a structure to store p-values
    p_values_ori_vs_shuf = struct();
    p_values_k_vs_kshuf = struct();

    % Loop over each subject
    for sub = subjects
        subj_field = sprintf('sub%02i', sub);
        
        % Loop over each k-value
        for k = k_values
            kval = sprintf('k%02i', k);
            
            % Get the percentile for this subject and k-value
            percentile_k = percentile_k_vs_kshuf.(subj_field).(kval);

            % Calculate the p-value
            p_value_k = 2 * min((1 - (percentile_k / 100)),(percentile_k / 100));

            % Store the p-value in the structure
            p_values_k_vs_kshuf.(subj_field).(kval) = p_value_k;
            

            % Loop over each brain region (lobe)
            for i = 1:length(lobes)
                lobename = lobes{i};
                
                % Get the percentile for this subject, k-value, and lobe
                percentile = percentile_ori_vs_shuf.(subj_field).(sprintf('%s_%s', kval, lobename));
                
                % Calculate the p-value
                p_value = 2 * min((1 - (percentile / 100)),(percentile / 100));
                
                % Store the p-value in the structure
                p_values_ori_vs_shuf.(subj_field).(sprintf('%s_%s', kval, lobename)) = p_value;
            end
        end
    end

    %% save the p-values
    save('../../results/p_values_k_vs_kshuf_new.mat', 'p_values_k_vs_kshuf');
    save('../../results/p_values_ori_vs_shuf_new.mat', 'p_values_ori_vs_shuf');


    %%

    %% Combine p values using Fisher's and Stouffer's methods
    % Initialize structures to store combined p-values
    combined_p_values_fisher = struct();
    combined_p_values_stouffer = struct();

    combined_p_values_fisher_k = struct();
    combined_p_values_stouffer_k = struct();



    % Loop over each k-value
    for k = k_values
        kval = sprintf('k%02i', k);

        % Collect p-values for all subjects for this k-value
        p_valuesk = [];
        for sub = subjects
            subj_field = sprintf('sub%02i', sub);
            p_valuesk = [p_valuesk; p_values_k_vs_kshuf.(subj_field).(kval)];
        end

        % Fisher's method to combine p-values
        chi2_statk = -2 * sum(log(p_valuesk));  % Chi-squared statistic
        dfk = 2 * length(p_valuesk);  % Degrees of freedom
        combined_p_value_fisher_k = 1 - chi2cdf(chi2_statk, dfk);  % Combined p-value

        % Stouffer's method to combine p-values
        z_scoresk = norminv(1 - p_valuesk);  % Convert p-values to z-scores
        z_combinedk = sum(z_scoresk) / sqrt(length(z_scoresk));  % Combined z-score
        combined_p_value_stouffer_k = 1 - normcdf(z_combinedk);  % Combined p-value

        % Store the combined p-values
        combined_p_values_fisher_k.(kval) = combined_p_value_fisher_k;
        combined_p_values_stouffer_k.(kval) = combined_p_value_stouffer_k;
        
        
        % Loop over each brain region (lobe)
        for i = 1:length(lobes)
            lobename = lobes{i};
            
            % Collect p-values for all subjects for this k-value and lobe
            p_values = [];
            for sub = subjects
                subj_field = sprintf('sub%02i', sub);
                p_values = [p_values; p_values_ori_vs_shuf.(subj_field).(sprintf('%s_%s', kval, lobename))];
            end
            
            % Fisher's method to combine p-values
            chi2_stat = -2 * sum(log(p_values));  % Chi-squared statistic
            df = 2 * length(p_values);  % Degrees of freedom
            combined_p_value_fisher = 1 - chi2cdf(chi2_stat, df);  % Combined p-value
            
            % Stouffer's method to combine p-values
            z_scores = norminv(1 - p_values);  % Convert p-values to z-scores
            z_combined = sum(z_scores) / sqrt(length(z_scores));  % Combined z-score
            combined_p_value_stouffer = 1 - normcdf(z_combined);  % Combined p-value
            
            % Store the combined p-values
            combined_p_values_fisher.(kval).(lobename) = combined_p_value_fisher;
            combined_p_values_stouffer.(kval).(lobename) = combined_p_value_stouffer;
        end
    end



    %% Save the results
    save('../../results/combined_p_values_fisher_k_new.mat', 'combined_p_values_fisher_k');
    save('../../results/combined_p_values_stouffer_k_new.mat', 'combined_p_values_stouffer_k');

    save('../../results/combined_p_values_fisher.mat', 'combined_p_values_fisher');
    save('../../results/combined_p_values_stouffer.mat', 'combined_p_values_stouffer');

    %% take out all the p values from all the combined p values and put them in a single array for plotting
    all_p_values_fisher = [];
    all_p_values_stouffer = [];

    for k = k_values
        kval = sprintf('k%02i', k);
        all_p_values_fisher = [all_p_values_fisher; combined_p_values_fisher_k.(kval)];
        all_p_values_stouffer = [all_p_values_stouffer; combined_p_values_stouffer_k.(kval)];
        for i = 1:length(lobes)
            lobename = lobes{i};
            all_p_values_fisher = [all_p_values_fisher; combined_p_values_fisher.(kval).(lobename)];
            all_p_values_stouffer = [all_p_values_stouffer; combined_p_values_stouffer.(kval).(lobename)];
        end
    end


    %% PLot all_p_values_fisher in a single line plot

    figure('Position', [100, 100, 800, 500], 'Color', 'white');

    % Plot the p-values with a thicker blue line
    plot(all_p_values_fisher, 'LineWidth', 2, 'Color', [0.2, 0.4, 0.8]);
    hold on;

    % Add only the 0.025 significance level with magenta dashed line
    plot([0, length(all_p_values_fisher)], [0.025, 0.025], '--', 'Color', [1, 0, 1], 'LineWidth', 2);

    % Add marker for original specification (first point)
    plot(1, all_p_values_fisher(1), 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'red');

    % Set axis limits
    xlim([0, length(all_p_values_fisher)]);
    ylim([-0.002, 0.025]); % Allow space below zero

    % Manually draw x-axis at y = -0.0008 (slightly below 0)
    plot([0, length(all_p_values_fisher)], [-0.0008, -0.0008], 'k-', 'LineWidth', 1.5);

    % Label the axes
    xlabel('Specifications(sorted by p-values)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('p-values', 'FontSize', 14, 'FontWeight', 'bold');

    % Add legend
    legend('p-values', '0.025 significance level', 'Original Specification', 'FontSize', 12);

    % Remove normal axes
    box off;
    ax = gca;
    ax.LineWidth = 2;
    ax.FontSize = 12;
    ax.XTickLabel = {};  % Hide original x-tick labels
    set(ax, 'XColor', 'none');  % Hide the x-axis line

    % Add custom x-ticks
    for i = [0, 10, 20, 30, 40, 50]
        text(i, -0.0015, num2str(i), 'HorizontalAlignment', 'center', 'FontSize', 12);
    end

    % Customize the y-tick marks
    ax.YTick = [0.00, 0.01, 0.02];
    ax.TickDir = 'out';

    hold off;

    print('../../results/all_p_values_fisher.png', '-dpng', '-r300');

    %% PLot all_p_values_stouffer in a single line plot

    figure('Position', [100, 100, 800, 500], 'Color', 'white');

    % Plot the p-values with a thicker blue line
    plot(all_p_values_stouffer, 'LineWidth', 2, 'Color', [0.2, 0.4, 0.8]);
    hold on;

    % Add only the 0.025 significance level with magenta dashed line
    plot([0, length(all_p_values_stouffer)], [0.025, 0.025], '--', 'Color', [1, 0, 1], 'LineWidth', 2);

    % Add marker for original specification (first point)
    plot(1, all_p_values_stouffer(1), 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'red');

    % Set axis limits
    xlim([0, length(all_p_values_stouffer)]);
    ylim([-0.002, 0.025]); % Allow space below zero

    % Manually draw x-axis at y = -0.0008 (slightly below 0)
    plot([0, length(all_p_values_stouffer)], [-0.0008, -0.0008], 'k-', 'LineWidth', 1.5);

    % Label the axes
    xlabel('Specifications(sorted by p-values)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('p-values', 'FontSize', 14, 'FontWeight', 'bold');

    % Add legend
    legend('p-values', '0.025 significance level', 'Original Specification', 'FontSize', 12);

    % Remove normal axes
    box off;
    ax = gca;
    ax.LineWidth = 2;
    ax.FontSize = 12;
    ax.XTickLabel = {};  % Hide original x-tick labels
    set(ax, 'XColor', 'none');  % Hide the x-axis line

    % Add custom x-ticks
    for i = [0, 10, 20, 30, 40, 50]
        text(i, -0.0015, num2str(i), 'HorizontalAlignment', 'center', 'FontSize', 12);
    end

    % Customize the y-tick marks
    ax.YTick = [0.00, 0.01, 0.02];
    ax.TickDir = 'out';

    hold off;

    print('../../results/all_p_values_stouffer.png', '-dpng', '-r300');
end