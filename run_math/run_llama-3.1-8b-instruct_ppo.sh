set -x

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

WORK_DIR=/opt/tiger/dwn-verl
MODEL=/mnt/bn/daiweinan-fuse/models/Llama-3.1-8B-Instruct
GPUS_PER_NODE=8
ACTOR_MICRO_BS=16
ROLLOUT_MICRO_BS=16
ROLLOUT_TP=8
REF_MICRO_BS=16
CRITIC_MICRO_BS=16

python3 -m verl.trainer.main_ppo \
    data.train_files=$WORK_DIR/run_math/math_ppo/train.parquet \
    data.val_files=$WORK_DIR/run_math/math_ppo/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=500 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$ACTOR_MICRO_BS \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$ROLLOUT_MICRO_BS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$REF_MICRO_BS \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.path=$MODEL \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size=$CRITIC_MICRO_BS \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.grad_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name='verl_example' \
    trainer.experiment_name='llama-3.1-8b-instruct_math_ppo' \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=15