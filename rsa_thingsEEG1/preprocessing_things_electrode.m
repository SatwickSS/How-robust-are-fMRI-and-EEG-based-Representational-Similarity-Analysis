function preprocess_eeg_data(partid, shuffles)
    % Combined preprocessing script for the THINGS EEG dataset
    % Input:
    %   partid - participant ID number
    %   shuffles - number of shuffles to generate (default: 10)

    if nargin < 2
        shuffles = 10;
    end

    %%add path
    addpath('../../CoSMoMVPA/mvpa')
    addpath('../../eeglab14_0_0b')

    %% launch EEGlab
    eeglab

    %% get files
    datapath = '../data'; % Change this to your data path
    savepath = datapath; % You can change this if you want to save to a different location

    % Create necessary directories
    mkdir(sprintf('%s/derivatives_ele/eeglab',datapath));
    mkdir(sprintf('%s/derivatives_ele/cosmomvpa',datapath));
    mkdir(sprintf('%s/derivatives_ele/epoched',datapath));

    % Create directories for shuffled data
    for nos=1:shuffles
        mkdir(sprintf('%s/lobes_cosmomvpa_shuffled/shuffle%02i',savepath,nos));
    end

    %% Process each participant
    fprintf('Starting Preprocessing for Participant:%02i\n',partid);
    % contfn = sprintf('%s/derivatives_ele/eeglab/sub-%02i_task-rsvp_continuous.set',datapath,partid);
    
    %% Load EEG file
    EEG_raw = pop_loadbv(sprintf('%s/sub-%02i/eeg/',datapath,partid),sprintf('sub-%02i_task-rsvp_eeg.vhdr',partid));
    EEG_raw = eeg_checkset(EEG_raw);
    EEG_raw.setname = partid;
    EEG_raw = eeg_checkset(EEG_raw);
    
    %% Re-reference
    if ismember(partid,[49 50]) %these were recorded with a 128 workspace and different ref, so remove the extra channels
        EEG_raw = pop_select(EEG_raw,'channel',1:63);
        EEG_raw = pop_chanedit(EEG_raw, 'append',1,'changefield',{2 'labels' 'FCz'},'setref',{'' 'FCz'});
        EEG_raw = pop_reref(EEG_raw, [],'refloc',struct('labels',{'FCz'},'type',{''},'theta',{0},'radius',{0.1278},'X',{0.3907},'Y',{0},'Z',{0.9205},'sph_theta',{0},'sph_phi',{67},'sph_radius',{1},'urchan',{[]},'ref',{''},'datachan',{0}));
    else
        EEG_raw = pop_chanedit(EEG_raw, 'append',1,'changefield',{2 'labels' 'Cz'},'setref',{'' 'Cz'});
        EEG_raw = pop_reref(EEG_raw, [],'refloc',struct('labels',{'Cz'},'type',{''},'theta',{0},'radius',{0},'X',{0},'Y',{0},'Z',{85},'sph_theta',{0},'sph_phi',{90},'sph_radius',{85},'urchan',{[]},'ref',{''},'datachan',{0}));
    end

    %% High pass filter
    EEG_raw = pop_eegfiltnew(EEG_raw, 0.1,[]);

    %% Low pass filter
    EEG_raw = pop_eegfiltnew(EEG_raw, [],100);

    %% Downsample
    EEG_cont = pop_resample(EEG_raw, 250);
    EEG_cont = eeg_checkset(EEG_cont);

    % Save continuous data
    % pop_saveset(EEG_cont,contfn);

    %% Add event info to events
    eventsfncsv = sprintf('%s/sub-%02i/eeg/sub-%02i_task-rsvp_events.csv',datapath,partid,partid);
    eventsfntsv = strrep(eventsfncsv,'.csv','.tsv');
    eventlist = readtable(eventsfncsv);

    idx = find(strcmp({EEG_cont.event.type},'E  1'));
    onset = vertcat(EEG_cont.event(idx).latency)*4-3;
    duration = 50*ones(size(onset));

    neweventlist = [table(onset,duration,'VariableNames',{'onset','duration'}) eventlist(1:numel(onset),:)];
    writetable(neweventlist,eventsfntsv,'filetype','text','Delimiter','\t');

    %% Create epochs
    % epochdata = sprintf('%s/derivatives_ele/epoched/sub-%02i_task-rsvp_continuous_epoched.set',datapath,partid);
    EEG_epoch = pop_epoch(EEG_cont, {'E  1'}, [-0.100 1.000]);
    EEG_epoch = eeg_checkset(EEG_epoch);
    % pop_saveset(EEG_epoch,epochdata); 
    
    %% Define lobe segregation
    frontal = [1 2 3 4 29 30 31 32 33 34 35 36 59 60 61 62 63]; %%AF,F, FP -> BLUE ->17
    central = [6 7 8 11 12 22 23 24 27 28 38 39 40 42 52 53 55 56 57 64]; %%FC, C, CP -> YELLOW ->21
    temporal = [5 9 10 21 25 26 37 41 54 58]; %%FT, T, TP -> GREEN ->10
    parietal_occipital = [13 14 15 16 17 18 19 20 43 44 45 46 47 48 49 50 51]; %%P, PO, O -> RED ->17

    lobe_index = {frontal,central,temporal,parietal_occipital};
    lobe_names = {'frontal','central','temporal','parietal_occipital'};
    
    %% Process original data - segregate into lobes and convert to cosmo
    lobes = struct();
    for l = 1:numel(lobe_index)
        lobes.(lobe_names{l}) = cosmo_flatten(permute(EEG_epoch.data(lobe_index{l},:,:),[3 1 2]),{'chan', 'time'}, {{EEG_epoch.chanlocs(lobe_index{l}).labels}, EEG_epoch.times},2);
        lobes.(lobe_names{l}).a.meeg=struct();
        lobes.(lobe_names{l}).sa = table2struct(eventlist,'ToScalar',true);
        cosmo_check_dataset(lobes.(lobe_names{l}),'meeg');
    end

    %% Save original lobes data
    fprintf('Saving Original Data for Participant:%02i\n',partid);
    savefile = sprintf('%s/derivatives_ele/cosmomvpa/sub-%02i_task-rsvp_cosmomvpa_lobes.mat',datapath,partid);
    save(savefile,'lobes','-v7.3');
    
    %% Process shuffled data
    for j=1:shuffles
        %% Shuffle to generate the null distribution
        fprintf('Starting Shuffle %02i | Participant: %02i.\n',j,partid);
        EEG_shuff = EEG_epoch;
        EEG_shuff.data = shuffle(EEG_shuff.data,3);
        
        %% Convert shuffled data to cosmo format
        lobes_shuffled = struct();
        for l = 1:numel(lobe_index)
            lobes_shuffled.(lobe_names{l}) = cosmo_flatten(permute(EEG_shuff.data(lobe_index{l},:,:),[3 1 2]),{'chan', 'time'}, {{EEG_shuff.chanlocs(lobe_index{l}).labels}, EEG_shuff.times},2);
            lobes_shuffled.(lobe_names{l}).a.meeg=struct();
            lobes_shuffled.(lobe_names{l}).sa = table2struct(eventlist,'ToScalar',true);
            cosmo_check_dataset(lobes_shuffled.(lobe_names{l}),'meeg');
        end
    
        %% Save shuffled data  
        fprintf('Saving Cosmo Shuffle %02i | Participant: %02i.\n',j,partid);
        savefile = sprintf('%s/lobes_cosmomvpa_shuffled/shuffle%02i/sub-%02i_task-rsvp_cosmomvpa_lobes_shuffle%02i.mat',savepath,j,partid,j); 
        save(savefile,'lobes','-v7.3');
    end
    
    fprintf('Finished processing Participant:%02i\n', partid);
    fprintf('All processing complete.\n');
end
