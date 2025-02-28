#!usr/bin/env bash

subject=01
nprocs=16
maxmem=80

#specify the directories

rawdatadir="/path/to/data"
derivsdir="/path/to/derivs_dir"
workdir="/path/to/workdir"
license_dir="/path/to/license_dir"

#run the fmriprep docker container

docker run -ti --rm \
  --memory="$maxmem""g" \
  -v "$rawdatadir":/data:ro \
  -v "$derivsdir":/out \
  -v "$workdir":/work \
  -v "$license_dir":/licensedir \
  nipreps/fmriprep:latest \
  --skip-bids-validation \
  --participant-label "$subject" \
  --output-spaces T1w func \
  --bold2t1w-dof 9 \
  --nprocs "$nprocs" --mem "$maxmem""GB" \
  --fs-license-file /licensedir/license.txt \
  /data /out participant -w /work



