set -x

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

WORK_DIR=/opt/tiger/dwn-verl
MODEL=hdfs://haruna/home/byte_data_seed/lf_lq/user/hongli/mathmodel/Meta-Llama-3.1-8B-Instruct
GPUS_PER_NODE=8
NNODES=8
ACTOR_MICRO_BS=$(( 8 * $NNODES ))
ROLLOUT_MICRO_BS=$(( 8 * $NNODES ))
ROLLOUT_TP=1
REF_MICRO_BS=$(( 8 * $NNODES ))
CRITIC_MICRO_BS=$(( 32 * $NNODES ))

python3 -m verl.trainer.main_ppo \
    data.train_files=$WORK_DIR/run_openai_math/openai_math_verl/train.parquet \
    data.val_files=$WORK_DIR/run_openai_math/openai_math_verl/test.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=500 \
    data.max_prompt_length=3000 \
    data.max_response_length=12288 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$ACTOR_MICRO_BS \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$ROLLOUT_MICRO_BS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$REF_MICRO_BS \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.path=$MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=$CRITIC_MICRO_BS \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.grad_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name='verl_example' \
    trainer.experiment_name=test_fork_${NNODES}nodes/openai_math_ppo_b1k_mb128_warm0 \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.default_local_dir='/mnt/bn/honglifish/model/verl/${trainer.project_name}/${trainer.experiment_name}' \
    trainer.default_hdfs_dir='hdfs://haruna/home/byte_data_seed/lf_lq/user/hongli/model/verl/${trainer.project_name}/${trainer.experiment_name}'