#!/usr/bin/env bash

subject=02
nprocs=16
maxmem=80 # 400

rawdatadir="/path/to/THINGS-data"
derivsdir="/path/to/preproc_derivative_poldracklab"
workdir="/path/to/preproc_workdir_poldracklab"
freesurfer_dir="/path/to/freesurfer"

docker run -ti --rm \
  --memory="$maxmem""g" \
  -v "$rawdatadir":/data:ro \
  -v "$derivsdir":/out \
  -v "$workdir":/work \
  -v "/path/to/licensedir":/licensedir \
  poldracklab/fmriprep:20.2.0 \
  --participant-label "$subject" \
  --fs-no-reconall \
  --fs-subjects-dir "$freesurfer_dir" \
  --output-spaces T1w func \
  --bold2t1w-dof 9 \
  --nprocs "$nprocs" --mem "$maxmem""GB" \
  --fs-license-file /licensedir/license.txt \
  --bids-filter-file /licensedir/filtercfg.json \
  -w /work /data /out participant
