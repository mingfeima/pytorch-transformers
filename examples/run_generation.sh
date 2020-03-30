###
###  reference: https://github.com/huggingface/transformers/tree/master/examples#language-generation
###

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";
export LD_PRELOAD=/home/mingfeim/packages/jemalloc-5.2.0/lib/libjemalloc.so


# by default the script use single socket, chage CORES if you want to use different number of cores
#CORES=`lscpu | grep Core | awk '{print $4}'`
CORES=12


SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
LAST_CORE=`expr $CORES - 1`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

export OMP_NUM_THREADS=$CORES
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"

echo -e "### using OMP_NUM_THREADS=$CORES"
echo -e "### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME"
echo -e "### using $PREFIX\n"


### fp32
#echo -e "### using fp32 model\n"
#$PREFIX python run_generation_20200320.py --model_type=gpt2 --model_name_or_path=gpt2

## profiler
#$PREFIX python run_generation_20200320.py --model_type=gpt2 --model_name_or_path=gpt2 --profile
#$PREFIX python -m cProfile -o output.pstats run_generation_20200320.py --model_type=gpt2 --model_name_or_path=gpt2

### int8
echo -e "### using int8 model\n"
$PREFIX python run_generation_20200320.py --model_type=gpt2 --model_name_or_path=gpt2 --quantize
