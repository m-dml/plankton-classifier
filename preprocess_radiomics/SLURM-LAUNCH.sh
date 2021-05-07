#!/bin/bash
#SBATCH --job-name=preprocess-plankton

### Modify this according to your Ray workload.
#SBATCH --nodes=3
#SBATCH --exclusive

#SBATCH --ntasks-per-node=1  # do not change

### Modify this according to your Ray workload.
#SBATCH --cpus-per-task=45
#SBATCH --time=8:00:00
#SBATCH --account=machnitz
#SBATCH --partition=pAll
#SBATCH --output=slurm_output/slurm-%j.out


################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

export RAY_ADDRESS=$ip_head
export  REDIS_PWD=$redis_password

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w $node_1 start-head.sh $ip $redis_password &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((  i=1; i<=$worker_num; i++ ))
do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i start-worker.sh $ip_head $redis_password &
  sleep 5
done
##############################################################################################

#### call your code below
srun /gpfs/home/machnitz/miniconda3/envs/plankton/bin/python preoprocess.py
exit
