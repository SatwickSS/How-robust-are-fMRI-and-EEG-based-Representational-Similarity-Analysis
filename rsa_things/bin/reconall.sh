#!/usr/bin/env bash

thingsmridir="/path/to/thingsmri_"
subject=02
nprocs=16

docker run -it --rm --mount \
  type=bind,source="${thingsmridir}",target=/thingsmri \
  surfrecon:latest \
  python /thingsmri/bids/code/things/mri/reconall.py "${subject}" "${nprocs}" "/thingsmri"
