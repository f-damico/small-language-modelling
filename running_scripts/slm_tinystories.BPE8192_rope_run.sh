#!/bin/bash


depth=12
T=512

P=$1

nh=$2
de=$((${nh}*64))

bs=$3
lr=$4
dt=$5
ep=$6

jobname=T${T}_P${P}_d${depth}_de${de}_nh${nh}_lr${lr}_bs${bs}_dt${dt}
echo ${jobname}

sed "s/_insert_name_/${jobname}/; s/_insert_T_/${T}/; s/_insert_P_/${P}/; s/_insert_de_/${de}/; s/_insert_nh_/${nh}/; s/_insert_depth_/${depth}/; s/_insert_lr_/${lr}/; s/_insert_bs_/${bs}/; s/_insert_dt_/${dt}/; s/_insert_ep_/${ep}/;" slm_tinystories.BPE8192_rope.VRG >> run_slm_tinystories.BPE8192_rope_${jobname}.sb

sbatch run_slm_tinystories.BPE8192_rope_${jobname}.sb
