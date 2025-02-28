#!usr/bin/env bash
# specify the variables
subject=01
nprocs=16
maxmem=80


#specify the directories
rawdatadir="/path/to/data"
derivsdir="/path/to/derivs_dir"
workdir="/path/to/workdir"
license_dir="/path/to/license_dir"


#Loop through the sessions
for i in {1..4}
do 
    #create a variable called session
    session=$(printf "filtercfg_ptest%02d.json" $i)
    #print the session
    echo "Running fmriprep for session :$session"

    #run the fmriprep docker script 
    if [ $i -eq 1 ]
    then 
        bash $(pwd)/fmriprep_cmdarg_script.sh $subject $nprocs $maxmem $rawdatadir $derivsdir $workdir $license_dir $session
    else
        bash $(pwd)/fmriprep_cmdarg_script_no_reconall.sh $subject $nprocs $maxmem $rawdatadir $derivsdir $workdir $license_dir $session
    fi
done