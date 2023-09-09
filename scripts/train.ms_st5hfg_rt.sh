#!/bin/sh


set -e

exec python train.py --config config_v3.16kHz.json --fine_tuning true --generator_only true \
	--checkpoint_path cp_ms_st5hfg_rt --summary_interval 25 \
	--validation_interval 1000 --checkpoint_interval 3000 "${@}"
