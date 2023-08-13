#!/bin/bash

# Define a list of arguments sets
argument_sets=(
    "dataset Cars gpu_id 0 batch_size 8 attn_heads 2 concat_heads True reduce_lr_bool True model_name scgnngatres_comb run_name run__attnh_2_concat_T wandb_log True"
    "dataset Cars gpu_id 0 batch_size 8 attn_heads 2 concat_heads False reduce_lr_bool True model_name scgnngatres_comb run_name run__attnh_2_concat_F wandb_log True"
    "dataset Cars gpu_id 0 batch_size 8 attn_heads 4 concat_heads True reduce_lr_bool True model_name scgnngatres_comb run_name run__attnh_4_concat_T wandb_log True"
    "dataset Cars gpu_id 0 batch_size 8 attn_heads 4 concat_heads False reduce_lr_bool True model_name scgnngatres_comb run_name run__attnh_4_concat_F wandb_log True"
    
    "dataset Cars gpu_id 0 batch_size 8 appnp_activation relu gat_activation relu reduce_lr_bool True model_name scgnngatres_comb run_name run__appnp_act_relu_gat_act_relu  wandb_log True"
    "dataset Cars gpu_id 0 batch_size 8 appnp_activation sigmoid gat_activation sigmoid reduce_lr_bool True model_name scgnngatres_comb run_name run__appnp_act_sigmd_gat_act_sigmd wandb_log True"
    
    "dataset Cars gpu_id 0 batch_size 8 l2_reg 2.5e-3 reduce_lr_bool True model_name scgnngatres_comb run_name run__l2_reg_2.5e-3 wandb_log True"
    "dataset Cars gpu_id 0 batch_size 8 l2_reg 2.5e-5 reduce_lr_bool True model_name scgnngatres_comb run_name  run__l2_reg_2.5e-5 wandb_log True"
)

# Define the number of tasks to run in each batch
tasks_per_batch=4

# Calculate the number of batches
num_batches=$(( (${#argument_sets[@]} + $tasks_per_batch - 1) / $tasks_per_batch ))

# Loop through the number of batches
for ((batch=1; batch<=$num_batches; batch++)); do
    # Calculate the start index for the current batch
    start_idx=$(( ($batch - 1) * $tasks_per_batch ))

    # Slice the argument sets for the current batch
    current_batch=("${argument_sets[@]:$start_idx:$tasks_per_batch}")

    # Generate prompts for the current batch
    prompts=()
    for args in "${current_batch[@]}"; do
        prompt="python ./SC_GNN/main.py $args"
        prompts+=("$prompt")
    done

    # Run generated prompts in parallel for the current batch
    parallel ::: "${prompts[@]}"

    # Print a message after the current batch is complete
    echo "Batch $batch is complete."

    # Sleep for a while before starting the next batch
    sleep 5  # Adjust as needed
done
