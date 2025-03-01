function make_rdms_electrode(subjectnr, k)
    if nargin < 2
        k = 1;
    end

    addpath('../../CoSMoMVPA/mvpa');
    % start cluster, give it a unique directory
    % starting a pool can fail when 2 procs are requesting simultaneous
    % thus try again after a second until success
    %% init pool
    pool=[];
    while isempty(pool) 
        try
            pc = parcluster('local');
            pc.JobStorageLocation=tempdir;
            pool=parpool(pc,48);
        catch err
            disp(err)
            delete(gcp('nocreate'));
            pause(1)
        end
    end
    nproc=cosmo_parallel_get_nproc_available();

    fprintf('Number of Processors Available: %i\n',nproc)

    %% get params
    defaults = struct();
    defaults.subject=subjectnr;
    opt = cosmo_structjoin(defaults);
    subjectnr = opt.subject;

    %% load data
    fn = sprintf('../../derivatives_ele/cosmomvpa/sub-%02i_task-rsvp_cosmomvpa_lobes.mat',subjectnr);
    fprintf('loading %s\n', fn); tic
    load(fn, 'lobes');
    fprintf('loading data finished in %i seconds\n', ceil(toc));

    %%  slice the dataset to only include the test stimuli
    lobe_names = {'frontal', 'central', 'temporal', 'parietal_occipital'};
    ds_lobes = struct();
    for i = 1:length(lobe_names)
        ds_lobes.(lobe_names{i}) = cosmo_slice(lobes.(lobe_names{i}), lobes.(lobe_names{i}).sa.isteststim > 0, 1);
    end

    %% prepare: set up targets and chunks for each lobe
    nh = struct();
    for i = 1:length(lobe_names)
        ds_lobes.(lobe_names{i}).sa.targets = 1 + ds_lobes.(lobe_names{i}).sa.teststimnumber;
        ds_lobes.(lobe_names{i}).sa.chunks = 1 + ds_lobes.(lobe_names{i}).sa.sequencenumber;
        nh.(lobe_names{i}) = cosmo_interval_neighborhood(ds_lobes.(lobe_names{i}), 'time', 'radius', 0);    
    end

    %% find all pairwise combinations
    ut = struct();
    combs = struct();
    uc = struct();

    for i = 1:length(lobe_names)
        ut.(lobe_names{i}) = unique(ds_lobes.(lobe_names{i}).sa.targets);
        combs.(lobe_names{i}) = combnk(ut.(lobe_names{i}), 2);
        uc.(lobe_names{i}) = unique(ds_lobes.(lobe_names{i}).sa.chunks);
    end


    %% prepare: set up targets and chunks for each lobe and find the items belonging to the exemplars

    target_idx = struct();
    for  i = 1:length(lobe_names)
        target_idx.(lobe_names{i}) = cell(1, length(ut.(lobe_names{i})));
        for j = 1:length(ut.(lobe_names{i}))
            target_idx.(lobe_names{i}){j} = find(ds_lobes.(lobe_names{i}).sa.targets == ut.(lobe_names{i})(j));
        end
    end

    test_chunk_idx = struct();
    for i = 1:length(lobe_names)
        test_chunk_idx.(lobe_names{i}) = cell(1, length(uc.(lobe_names{i})));
        for j = 1:length(uc.(lobe_names{i}))
            test_chunk_idx.(lobe_names{i}){j} = find(ds_lobes.(lobe_names{i}).sa.chunks == uc.(lobe_names{i})(j));
        end
    end


    %% make blocks for parfor loop
    step = ceil(length(combs.(lobe_names{1}))/nproc);
    s = 1:step:length(combs.(lobe_names{1}));
    comb_blocks = cell(1,length(s));
    for b = 1:nproc
        comb_blocks{b} = combs.(lobe_names{1})(s(b):min(s(b)+step-1,length(combs.(lobe_names{1}))),:);
    end

    %arguments for searchlight and crossvalidation
    ma = struct();
    ma.classifier = @cosmo_classify_lda;
    ma.output = 'accuracy';
    ma.check_partitions = false;
    ma.nproc = 1;
    ma.progress = 0;
    ma.partitions = struct();

    result_map_cell = struct();
    for i = 1:length(lobe_names)
        worker_opt_cell = cell(1,nproc);
        for p=1:nproc
            worker_opt=struct();
            worker_opt.ds=ds_lobes.(lobe_names{i});
            worker_opt.k=k;
            worker_opt.ma = ma;
            worker_opt.uc = uc.(lobe_names{i});
            worker_opt.worker_id=p;
            worker_opt.nproc=nproc;
            worker_opt.nh=nh.(lobe_names{i});
            worker_opt.combs = comb_blocks{p};
            worker_opt.target_idx = target_idx.(lobe_names{i});
            worker_opt.test_chunk_idx = test_chunk_idx.(lobe_names{i});
            worker_opt_cell{p}=worker_opt;
        end
        result_map_cell.(lobe_names{i}) = cosmo_parcellfun(nproc,@run_block_with_worker,...
                                        worker_opt_cell,'UniformOutput',false);
    end

    %% cat the results
    res = struct();
    for i = 1:length(lobe_names)
        res.(lobe_names{i}) = cosmo_stack(result_map_cell.(lobe_names{i}));
        res.(lobe_names{i}).sa.target1stim = ds_lobes.(lobe_names{i}).sa.stim(res.(lobe_names{i}).sa.target1);
        res.(lobe_names{i}).sa.target2stim = ds_lobes.(lobe_names{i}).sa.stim(res.(lobe_names{i}).sa.target2);
    end

    %% save
    fprintf('Saving...');tic
    outfn = sprintf('../../derivatives/lobes/sub-%02i_rdm_test_images_lobes_k%02i.mat',subjectnr,k);
    save(outfn,'res','-v7.3')
    fprintf('Saving finished in %i seconds\n',ceil(toc))
    quit
end

function res_block = run_block_with_worker(worker_opt)
    ds=worker_opt.ds;
    nh=worker_opt.nh;
    k=worker_opt.k;
    ma=worker_opt.ma;
    uc=worker_opt.uc;
    target_idx=worker_opt.target_idx;
    test_chunk_idx=worker_opt.test_chunk_idx;
    worker_id=worker_opt.worker_id;
    nproc=worker_opt.nproc;
    combs=worker_opt.combs;
    res_cell = cell(1,length(combs));
    cc=clock();mm='';
    for i=1:length(combs)
        fprintf('Processing block %i/%i | k =%02i\n', i, length(combs),k);
        idx_ex = [target_idx{combs(i,1)}; target_idx{combs(i,2)}];
        
        nck = nchoosek(1:length(uc), k);
        [ma.partitions.train_indices,ma.partitions.test_indices] = deal(cell(1,length(nck)));
        
        for j = 1:size(nck,1)
            ma.partitions.train_indices{j}=idx_ex;
            ma.partitions.test_indices{j}=idx_ex;
            for l = 1:size(nck,2)
                ma.partitions.train_indices{j} = setdiff(ma.partitions.train_indices{j},test_chunk_idx{nck(j,l)});
            end
            ma.partitions.test_indices{j}=setdiff(ma.partitions.test_indices{j},ma.partitions.train_indices{j});
        end
        res_cell{i} = cosmo_searchlight(ds,nh,@cosmo_crossvalidation_measure,ma);
        res_cell{i}.sa.target1 = combs(i,1);
        res_cell{i}.sa.target2 = combs(i,2);
        if ~mod(i,10)
            mm=cosmo_show_progress(cc,i/length(combs),sprintf('%i/%i for worker %i/%i\n',i,length(combs),worker_id,nproc),mm);
        end
    end
    res_block = cosmo_stack(res_cell);
end
