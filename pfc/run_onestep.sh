#!/bin/bash
# (PFC) Predicción a 1 paso ("teacher forcing") para DualFluidNet y DMCF + render coloreado por error.
# En cada frame el modelo recibe el estado REAL del simulador y predice el siguiente. NO es una simulación:
# muestra el error local del modelo. Se presenta junto al rollout normal (run_final.sh).
#   bash /workspace/DualFluidNet/pfc/run_onestep.sh
set -e
FRAMES=${FRAMES:-60}
SCENES=${SCENES:-"sim_0201 sim_0202"}
ERR_MAX_MM=${ERR_MAX_MM:-10}
DATA=/workspace/datasets/ours_6k_box_data
RUNS=/workspace/runs
RES=/workspace/results/onestep
DFN=${DFN:-/workspace/DualFluidNet}
DMCF=${DMCF:-/workspace/DMCF}
NSC=$(echo $SCENES | wc -w)
mkdir -p $RES/logs $RES/renders

DFN_CKPT=$(ls -v $RUNS/dfn_final/train_network_torch_runpod_6k/checkpoints/ckpt-*.pt | tail -1)
DMCF_CKPT=$(ls -v $RUNS/dmcf_final/logs/SymNet_CConvData3D_3d/checkpoint/ckpt-*.index | tail -1); DMCF_CKPT=${DMCF_CKPT%.index}
DFN_IT=$(basename $DFN_CKPT .pt | sed 's/ckpt-//'); DMCF_EP=$(basename $DMCF_CKPT | sed 's/ckpt-//')

source /workspace/miniforge3/bin/activate dfn
cd $DFN
echo "== [1/4] DFN: predicción a 1 paso"
for s in $SCENES; do
  python pfc/rollout_dfn.py --one_step --ckpt $DFN_CKPT --data_dir $DATA/valid --scene $s --frames $FRAMES \
    --out $RES/dfn_$s.npz > $RES/logs/dfn_$s.log 2>&1
done
conda deactivate

source /workspace/miniforge3/bin/activate dmcf
cd $DMCF
echo "== [2/4] DMCF: predicción a 1 paso (entrenado y oficial)"
for pair in "ours:$DMCF_CKPT" "official:checkpoints/Liquid3d/ckpt"; do
  name=${pair%%:*}; ckpt=${pair#*:}
  python pfc_onestep.py --cfg_file configs/Liquid3d.yml --dataset_path $DATA --ckpt_path $ckpt --split test \
    --pipeline.data_generator.test.time_end $((FRAMES+1)) --pipeline.data_generator.test.cnt $NSC \
    --pipeline.main_log_dir $RUNS/dmcf_onestep_$name/logs --pipeline.train_sum_dir $RUNS/dmcf_onestep_$name/train_log \
    --pipeline.output_dir $RUNS/dmcf_onestep_$name/output > $RES/logs/dmcf_$name.log 2>&1
done
conda deactivate

source /workspace/miniforge3/bin/activate dfn
cd $DFN
echo "== [3/4] Resumen"
ARGS=""; i=0
for s in $SCENES; do
  idx=$(printf "%04d" $i)
  ours=$(ls $RUNS/dmcf_onestep_ours/output/SymNet_CConvData3D_3d/visual/$idx/*.hdf5 | head -1)
  off=$(ls $RUNS/dmcf_onestep_official/output/SymNet_CConvData3D_3d/visual/$idx/*.hdf5 | head -1)
  ARGS="$ARGS --rollout DualFluidNet_${DFN_IT}it $s $RES/dfn_$s.npz --rollout DMCF_ep${DMCF_EP} $s $ours --rollout DMCF_oficial $s $off"
  i=$((i+1))
done
python pfc/summarize.py $ARGS --out_dir $RES | tee $RES/logs/summarize.log
echo "== [4/4] Render coloreado por error"
i=0
for s in $SCENES; do
  idx=$(printf "%04d" $i)
  ours=$(ls $RUNS/dmcf_onestep_ours/output/SymNet_CConvData3D_3d/visual/$idx/*.hdf5 | head -1)
  off=$(ls $RUNS/dmcf_onestep_official/output/SymNet_CConvData3D_3d/visual/$idx/*.hdf5 | head -1)
  python pfc/render.py --inputs $RES/dfn_$s.npz $ours $off \
    --labels "DualFluidNet (${DFN_IT} it)" "DMCF (ep ${DMCF_EP})" "DMCF oficial" \
    --out_dir $RES/renders/$s --every 2 --max_frames $FRAMES --color_error $ERR_MAX_MM \
    --title "predicción a 1 paso desde el estado real (no es rollout)"
  i=$((i+1))
done
cd /workspace && tar czf results_onestep.tar.gz results/onestep
echo "LISTO -> /workspace/results_onestep.tar.gz"
