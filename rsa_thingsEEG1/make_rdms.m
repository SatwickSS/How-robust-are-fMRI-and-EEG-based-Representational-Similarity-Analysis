function make_rdms(subjectnr, k)
    % by default k=1
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

    %% load data
    fn = sprintf('../../derivatives/cosmomvpa/original/sub-%02i_task-rsvp_cosmomvpa.mat',subjectnr);
    fprintf('loading %s\n',fn);tic
    load(fn,'ds')
    fprintf('loading data finished in %i seconds\n',ceil(toc))
    data_test = cosmo_slice(ds,ds.sa.isteststim>0,1);
    %The dataset is sliced to include only the test stimuli.

    ds = data_test;
    %% prepare: set up targets and chunks
    ds.sa.targets = 1+ds.sa.teststimnumber; %(add one for matlab)
    %"targets" refer to the labels or categories that you want to predict or classify. 
    ds.sa.chunks = 1+ds.sa.sequencenumber; %(add one for matlab)
    %Chunks represent independent groups of data that are not supposed to be mixed together during cross-validation or analysis.
    nh = cosmo_interval_neighborhood(ds,'time','radius',0);
    %create a neighborhood structure that defines a "temporal" or "time-based" radius around each sample.

    %% find all pairwise combinations
    ut = unique(ds.sa.targets);
    combs = combnk(ut,2);
    %ut contains unique target values. combs is a matrix where each row represents a unique pairwise combination of the target values.

    % all chunks to leave out
    uc = unique(ds.sa.chunks);

    % find the items belonging to the exemplars
    target_idx = cell(1,length(ut));
    for j=1:length(ut)
        target_idx{j} = find(ds.sa.targets==ut(j)); 
    end

    % for each chunk, find items belonging to the test set
    for j=1:length(uc)
        test_chunk_idx{j} = find(ds.sa.chunks==uc(j));
    end

    %% make blocks for parfor loop
    step = ceil(length(combs)/nproc);
    s = 1:step:length(combs);
    comb_blocks = cell(1,length(s));
    for b = 1:nproc
        comb_blocks{b} = combs(s(b):min(s(b)+step-1,length(combs)),:);
    end

    %arguments for searchlight and crossvalidation
    ma = struct();
    ma.classifier = @cosmo_classify_lda;
    ma.output = 'accuracy';
    ma.check_partitions = false;
    ma.nproc = 1;
    ma.progress = 0;
    ma.partitions = struct();

    % set options for each worker process
    worker_opt_cell = cell(1,nproc);
    for p=1:nproc
        worker_opt=struct();
        worker_opt.ds=ds;
        worker_opt.k=k;
        worker_opt.ma = ma;
        worker_opt.uc = uc;
        worker_opt.worker_id=p;
        worker_opt.nproc=nproc;
        worker_opt.nh=nh;
        worker_opt.combs = comb_blocks{p};
        worker_opt.target_idx = target_idx;
        worker_opt.test_chunk_idx = test_chunk_idx;
        worker_opt_cell{p}=worker_opt;
    end

    %% run the workers
    result_map_cell=cosmo_parcellfun(nproc,@run_block_with_worker,...
                                    worker_opt_cell,'UniformOutput',false);
    %% cat the results
    res=cosmo_stack(result_map_cell);
    res.sa.target1stim = ds.sa.stim(res.sa.target1);
    res.sa.target2stim = ds.sa.stim(res.sa.target2);

    %% save
    fprintf('Saving...');tic
    outfn = sprintf('../../derivatives/rdms/sub-%02i_rdm_test_images_k%02i.mat',subjectnr,k);
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
        fprintf('Processing block %i/%i | k =%02i', i, length(combs),k);
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
