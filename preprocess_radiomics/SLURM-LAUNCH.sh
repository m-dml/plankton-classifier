#!/bin/bash
#SBATCH --job-name=preprocess-plankton

### Modify this according to your Ray workload.
#SBATCH --nodes=20
#SBATCH --exclusive

#SBATCH --tasks-per-node=1  # do not change

### Modify this according to your Ray workload.
#SBATCH --cpus-per-task=42
#SBATCH --time=8:00:00
#SBATCH --account=machnitz
#SBATCH --partition=pAll
#SBATCH --exclusive
#SBATCH --output=slurm_output/slurm-%j.out

## create SLURM output directory if it doesn't exist
mkdir -p slurm_output

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

# start head node
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --block &


# start worker
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --block &
    sleep 5
done

echo "Running the pythone code..."
srun /gpfs/home/machnitz/miniconda3/envs/plankton/bin/python preoprocess.py