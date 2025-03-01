function preprocessing_things(partid, shuffles)
    % Combined preprocessing script for the THINGS EEG dataset
    % Input:
    %   partid - participant ID number
    %   shuffles - number of shuffles to generate (default: 10)
    
    if nargin < 2
        shuffles = 10;
    end
    
    %% Add necessary paths
    addpath('../../CoSMoMVPA/mvpa')
    addpath('../../eeglab14_0_0b')
    
    %% Launch EEGlab
    eeglab 
    
    %% Set up directories
    datapath = '../data'; % Change this to your data path
    savepath = datapath; % You can change this if you want to save to a different location
    
    % Create necessary directories
    mkdir(sprintf('%s/derivatives/eeglab', datapath));
    mkdir(sprintf('%s/derivatives/eeglab_shuffle', datapath));
    mkdir(sprintf('%s/derivatives/cosmomvpa/original', savepath));
    
    % Create shuffle directories
    for nos = 1:shuffles
        mkdir(sprintf('%s/derivatives/cosmomvpa_shuffled/shuffle%02i', savepath, nos));
        mkdir(sprintf('%s/derivatives/eeglab_shuffle/shuffle%02i', datapath, nos));
    end
    
    fprintf('Starting Preprocessing for Participant: %02i\n', partid);
    
    %% Define filenames
    contfn = sprintf('%s/derivatives/eeglab/sub-%02i_task-rsvp_continuous.set', datapath, partid);
    epochdata = sprintf('%s/derivatives/eeglab/sub-%02i_task-rsvp_continuous_epoched.set', datapath, partid); %-
    
    %% Load or preprocess continuous data
    if isfile(contfn)
        fprintf('Using existing continuous data file: %s\n', contfn);
        EEG_cont = pop_loadset(contfn);
    else
        % Load raw EEG file
        EEG_raw = pop_loadbv(sprintf('%s/sub-%02i/eeg/', datapath, partid), sprintf('sub-%02i_task-rsvp_eeg.vhdr', partid));
        EEG_raw = eeg_checkset(EEG_raw);
        EEG_raw.setname = num2str(partid);
        EEG_raw = eeg_checkset(EEG_raw);
        
        % Re-reference
        if ismember(partid,[49 50]) %these were recorded with a 128 workspace and different ref, so remove the extra channels
            EEG_raw = pop_select(EEG_raw,'channel',1:63);
            EEG_raw = pop_chanedit(EEG_raw, 'append',1,'changefield',{2 'labels' 'FCz'},'setref',{'' 'FCz'});
            EEG_raw = pop_reref(EEG_raw, [],'refloc',struct('labels',{'FCz'},'type',{''},'theta',{0},'radius',{0.1278},'X',{0.3907},'Y',{0},'Z',{0.9205},'sph_theta',{0},'sph_phi',{67},'sph_radius',{1},'urchan',{[]},'ref',{''},'datachan',{0}));
        else
            EEG_raw = pop_chanedit(EEG_raw, 'append',1,'changefield',{2 'labels' 'Cz'},'setref',{'' 'Cz'});
            EEG_raw = pop_reref(EEG_raw, [],'refloc',struct('labels',{'Cz'},'type',{''},'theta',{0},'radius',{0},'X',{0},'Y',{0},'Z',{85},'sph_theta',{0},'sph_phi',{90},'sph_radius',{85},'urchan',{[]},'ref',{''},'datachan',{0}));
        end
        
        % High pass filter
        EEG_raw = pop_eegfiltnew(EEG_raw, 0.1, []);
        
        % Low pass filter
        EEG_raw = pop_eegfiltnew(EEG_raw, [], 100);
        
        % Downsample
        EEG_cont = pop_resample(EEG_raw, 250);
        EEG_cont = eeg_checkset(EEG_cont);
        
        % Save continuous data
        pop_saveset(EEG_cont, contfn);
    end
    
    %% Add event information to events
    eventsfncsv = sprintf('%s/sub-%02i/eeg/sub-%02i_task-rsvp_events.csv', datapath, partid, partid);
    eventsfntsv = strrep(eventsfncsv, '.csv', '.tsv');
    eventlist = readtable(eventsfncsv);
    
    idx = find(strcmp({EEG_cont.event.type}, 'E  1'));
    onset = vertcat(EEG_cont.event(idx).latency) * 4 - 3;
    duration = 50 * ones(size(onset)); 
    
    neweventlist = [table(onset, duration, 'VariableNames', {'onset', 'duration'}) eventlist(1:numel(onset), :)];
    writetable(neweventlist, eventsfntsv, 'filetype', 'text', 'Delimiter', '\t');
    
    %% Create epochs
    EEG_epoch = pop_epoch(EEG_cont, {'E  1'}, [-0.100 1.000]);
    EEG_epoch = eeg_checkset(EEG_epoch);
    pop_saveset(EEG_epoch, epochdata); % Save epoched data
    
    %% Convert to CoSMoMVPA format (original data)
    ds = cosmo_flatten(permute(EEG_epoch.data, [3 1 2]), {'chan', 'time'}, {{EEG_epoch.chanlocs.labels}, EEG_epoch.times}, 2);
    ds.a.meeg = struct(); 
    ds.sa = table2struct(eventlist, 'ToScalar', true);
    cosmo_check_dataset(ds, 'meeg');
    
    %% Save original dataset
    fprintf('Saving Original CoSMoMVPA Dataset for Participant: %02i\n', partid);
    savefile = sprintf('%s/derivatives/cosmomvpa/original/sub-%02i_task-rsvp_cosmomvpa.mat', savepath, partid);
    save(savefile, 'ds', '-v7.3');
    
    %% Generate and save shuffled datasets
    for j = 1:shuffles
        % Shuffle epochs to generate null distribution
        EEG_shuff = EEG_epoch;
        EEG_shuff.data = shuffle(EEG_shuff.data, 3);
        
        % Save shuffled EEGLAB dataset
        shuf_eeglab = sprintf('%s/derivatives/eeglab_shuffle/shuffle%02i/sub-%02i_task-rsvp_continuous_shuffle%02i.set', datapath, j, partid, j);
        pop_saveset(EEG_shuff, shuf_eeglab);
        
        % Convert shuffled data to CoSMoMVPA format
        ds_shuf = cosmo_flatten(permute(EEG_shuff.data, [3 1 2]), {'chan', 'time'}, {{EEG_shuff.chanlocs.labels}, EEG_shuff.times}, 2);
        ds_shuf.a.meeg = struct();
        ds_shuf.sa = table2struct(eventlist, 'ToScalar', true);
        cosmo_check_dataset(ds_shuf, 'meeg');
        
        % Save shuffled CoSMoMVPA dataset
        fprintf('Saving Shuffled CoSMoMVPA Dataset %02i for Participant: %02i\n', j, partid);
        savefile = sprintf('%s/derivatives/cosmomvpa_shuffled/shuffle%02i/sub-%02i_task-rsvp_cosmomvpa_shuffle%02i.mat', savepath, j, partid, j);
        save(savefile, 'ds_shuf', '-v7.3');
    end
    
    fprintf('Finished preprocessing for Participant: %02i\n', partid);
end

