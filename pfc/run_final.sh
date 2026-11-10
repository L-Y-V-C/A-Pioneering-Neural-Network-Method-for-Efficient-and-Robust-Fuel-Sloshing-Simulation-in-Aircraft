#!/bin/bash
# Pipeline de resultados finales (PFC). Ejecutar en el pod con la GPU libre (entrenamientos terminados):
#   bash /workspace/DualFluidNet/pfc/run_final.sh
# Variables opcionales: FRAMES (default 60), SCENES ("sim_0201 sim_0202"), DMCF_VALID_CNT (default 3)
set -e
FRAMES=${FRAMES:-60}
SCENES=${SCENES:-"sim_0201 sim_0202"}
DMCF_VALID_CNT=${DMCF_VALID_CNT:-3}
START_STEP=${START_STEP:-1}  # para retomar: START_STEP=4 bash run_final.sh
DATA=/workspace/datasets/ours_6k_box_data
RUNS=/workspace/runs
RES=/workspace/results
DFN=/workspace/DualFluidNet
DMCF=/workspace/DMCF
NSC=$(echo $SCENES | wc -w)
mkdir -p $RES/rollouts $RES/curves $RES/renders $RES/logs

DFN_CKPT=$(ls -v $RUNS/dfn_final/train_network_torch_runpod_6k/checkpoints/ckpt-*.pt | tail -1)
DMCF_CKPT=$(ls -v $RUNS/dmcf_final/logs/SymNet_CConvData3D_3d/checkpoint/ckpt-*.index | tail -1); DMCF_CKPT=${DMCF_CKPT%.index}
echo "DFN_CKPT=$DFN_CKPT"; echo "DMCF_CKPT=$DMCF_CKPT"
DFN_IT=$(basename $DFN_CKPT .pt | sed 's/ckpt-//')
DMCF_EP=$(basename $DMCF_CKPT | sed 's/ckpt-//')

# ---------------- DualFluidNet ----------------
source /workspace/miniforge3/bin/activate dfn
cd $DFN
if [ $START_STEP -le 1 ]; then
echo "== [1/6] DFN: evaluación n-step en valid"
python pfc/eval_dfn.py --ckpt $DFN_CKPT --data_dir $DATA --out $RES/dfn_eval.json > $RES/logs/dfn_eval.log 2>&1
fi
if [ $START_STEP -le 2 ]; then
echo "== [2/6] DFN: rollouts ($FRAMES frames)"
for s in $SCENES; do
  python pfc/rollout_dfn.py --ckpt $DFN_CKPT --data_dir $DATA/valid --scene $s --frames $FRAMES \
    --out $RES/rollouts/dfn_$s.npz > $RES/logs/dfn_rollout_$s.log 2>&1
done
fi
conda deactivate

# ---------------- DMCF ----------------
source /workspace/miniforge3/bin/activate dmcf
cd $DMCF
run_dmcf() {  # $1=nombre $2=ckpt $3=split $4.. extra
  local name=$1 ckpt=$2 split=$3; shift 3
  python run_pipeline.py --cfg_file configs/Liquid3d.yml --dataset_path $DATA --ckpt_path $ckpt --split $split \
    --pipeline.main_log_dir $RUNS/$name/logs --pipeline.train_sum_dir $RUNS/$name/train_log \
    --pipeline.output_dir $RUNS/$name/output "$@" > $RES/logs/$name.log 2>&1
}
if [ $START_STEP -le 3 ]; then
echo "== [3/6] DMCF: rollouts de test (entrenado y oficial)"
run_dmcf dmcf_eval_ours $DMCF_CKPT test --pipeline.data_generator.test.time_end $((FRAMES+1)) --pipeline.data_generator.test.cnt $NSC
run_dmcf dmcf_eval_official checkpoints/Liquid3d/ckpt test --pipeline.data_generator.test.time_end $((FRAMES+1)) --pipeline.data_generator.test.cnt $NSC
fi
echo "== [4/6] DMCF: tiempo de inferencia (1 escena) y métricas de su paper en valid ($DMCF_VALID_CNT escenas)"
[ -f $RES/logs/dmcf_timing.log ] || run_dmcf dmcf_timing $DMCF_CKPT test --pipeline.data_generator.test.time_end 31 --pipeline.data_generator.test.cnt 1
# tqdm muestra "X s/it" (lento) o "X it/s" (rápido): se aceptan ambos formatos
RATE=$(tr '\r' '\n' < $RES/logs/dmcf_timing.log | grep -oE 'rollout: 100%.*' | tail -1 | grep -oE '[0-9.]+(s/it|it/s)' | tail -1)
DMCF_MS=$(python -c "r='$RATE'; v=float(r.split('s/it')[0].split('it/s')[0]); print(v*1000 if r.endswith('s/it') else 1000.0/v)")
echo "DMCF ms/frame: $DMCF_MS ($RATE)"
run_dmcf dmcf_valid_ours $DMCF_CKPT valid --pipeline.data_generator.valid.cnt $DMCF_VALID_CNT
run_dmcf dmcf_valid_official checkpoints/Liquid3d/ckpt valid --pipeline.data_generator.valid.cnt $DMCF_VALID_CNT
for n in dmcf_valid_ours dmcf_valid_official; do
  tr '\r' '\n' < $RES/logs/$n.log | grep -E "simulator - [0-9]+ - |validation of epoch" > $RES/${n}_metrics.txt || true
done
conda deactivate

# ---------------- Resumen, render y curvas ----------------
source /workspace/miniforge3/bin/activate dfn
cd $DFN
echo "== [5/6] Resumen de métricas"
ARGS=""
i=0
for s in $SCENES; do
  idx=$(printf "%04d" $i)
  ours=$(ls $RUNS/dmcf_eval_ours/output/SymNet_CConvData3D_3d/visual/$idx/*.hdf5 | head -1)
  off=$(ls $RUNS/dmcf_eval_official/output/SymNet_CConvData3D_3d/visual/$idx/*.hdf5 | head -1)
  ARGS="$ARGS --rollout DualFluidNet_${DFN_IT}it $s $RES/rollouts/dfn_$s.npz"
  ARGS="$ARGS --rollout DMCF_ep${DMCF_EP} $s $ours --rollout DMCF_oficial $s $off"
  i=$((i+1))
done
python pfc/summarize.py $ARGS --timing DMCF_ep${DMCF_EP} $DMCF_MS --timing DMCF_oficial $DMCF_MS \
  --extra $RES/dfn_eval.json --out_dir $RES | tee $RES/logs/summarize.log
echo "== [6/6] Render 3D + GIF y curvas de entrenamiento"
i=0
for s in $SCENES; do
  idx=$(printf "%04d" $i)
  ours=$(ls $RUNS/dmcf_eval_ours/output/SymNet_CConvData3D_3d/visual/$idx/*.hdf5 | head -1)
  off=$(ls $RUNS/dmcf_eval_official/output/SymNet_CConvData3D_3d/visual/$idx/*.hdf5 | head -1)
  python pfc/render.py --inputs $RES/rollouts/dfn_$s.npz $ours $off \
    --labels "DualFluidNet (${DFN_IT} it)" "DMCF (ep ${DMCF_EP})" "DMCF oficial" \
    --out_dir $RES/renders/$s --every 2 --max_frames $FRAMES
  i=$((i+1))
done
python pfc/plot_logs.py --logdir $RUNS/dfn_final --name DualFluidNet --out_dir $RES/curves
python pfc/plot_logs.py --logdir $RUNS/dmcf_final/train_log --name DMCF --out_dir $RES/curves
cp $RUNS/dfn_final/train*.log $RUNS/dmcf_final/train.log $RES/logs/ 2>/dev/null || true
cd /workspace && tar czf results.tar.gz results
echo "LISTO -> /workspace/results.tar.gz"
