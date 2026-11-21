# usage: try.sh TAG TRAIN "fit args" "apply args"
M=$(dirname $0); TAG=$1
python $M/fit_rs.py --train $2 --test preds/rs/practice.npz $3 --out preds/rs/m_$TAG.pkl | tail -1
python $M/pipeline_rs.py --mode apply --videos ../test_sample_clean --dets preds/hf_skyguard --model preds/rs/m_$TAG.pkl --out preds/rs/a_$TAG $4 | tail -1
echo "== $TAG"; python ~/Downloads/evaluation.py --gt_dir ../test_sample_clean --pred_dir preds/rs/a_$TAG | sed -n 6,9p
