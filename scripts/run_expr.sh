export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cudnn8.0-11.0/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cudnn8.0-11.0/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cudnn8.0-11.0/lib64:$LIBRARY_PATH
source activate research

for w in 0.6 0.62 0.64 0.7 0.72 0.74 0.76 0.8 0.82 0.84 0.86 0.9 0.92 0.94 0.96
  do
    echo "Weight" $w
    python /home/trduong/Data/fairCE/src/run_flow.py --weight $w --data_name simple_bn
    python /home/trduong/Data/fairCE/src/run_flow.py --weight $w --data_name adult
  done

python /home/trduong/Data/fairCE/src/run_gs.py --weight $w --data_name adult
python /home/trduong/Data/fairCE/src/run_gs.py --weight $w --data_name adult

