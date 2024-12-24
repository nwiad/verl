# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Callable, Type, Tuple, Union

from omegaconf import OmegaConf, open_dict
import numpy as np
from codetiming import Timer

from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl import DataProto
from verl.trainer.ppo import core_algos

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # Due to the Ray issue, we can only support max_colocate_count=1 for now.
            # This means that each GPU can only have one process.
            # We can support max_colocate > 1 when applying this pull request: https://github.com/ray-project/ray/pull/44385
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]] # dwn: map role to resource pool


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: Union[DataProto, list[DataProto]], gamma=1.0, lam=1.0, adv_estimator='gae', group_size=1):
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards'] # dwn: token_level_scores - beta * kld
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'norm_os':
        # dwn: batch size is bs * gs
        responses = data.batch['responses']
        response_length = responses.size(1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:] # dwn: bs * gs, response_length)
        token_level_scores = data.batch['token_level_scores'] # dwn: (bs * gs, response_length), do not apply kl penalty
        advantages = core_algos.compute_norm_os_advantage(token_level_scores=token_level_scores,
                                                          response_length=response_length,
                                                          eos_mask=response_mask,
                                                          group_size=group_size)
        data.batch['advantages'] = advantages # dwn: (bs * gs, response_length)
    elif adv_estimator == 'norm_ps':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def compute_data_metrics(batch):
    import wandb
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)

    response_length = batch.batch['responses'].shape[-1]

    advantages = batch.batch['advantages']
    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length_int = response_mask.sum(-1) # for eos_relative compute
    response_length = response_length_int.float()  # (batch_size,)
    
    metrics = {
        # score
        'critic/score/mean': torch.mean(sequence_score).detach().item(),
        'critic/score/max': torch.max(sequence_score).detach().item(),
        'critic/score/min': torch.min(sequence_score).detach().item(),
        'critic/score/hist': wandb.Histogram(sequence_score.detach()),
        # adv
        'critic/advantages/mean': masked_mean(advantages, response_mask).detach().item(),
        'critic/advantages/max': torch.max(advantages[response_mask.bool()]).detach().item(), #### CRITIC!!!
        'critic/advantages/min': torch.min(advantages[response_mask.bool()]).detach().item(),
        # response length
        'response_length/eos_adv_mean': torch.mean(advantages.gather(1, (response_length - 1).long().unsqueeze(1))).detach().item(),
        'response_length/mean': torch.mean(response_length).detach().item(),
        'response_length/max': torch.max(response_length).detach().item(),
        'response_length/min': torch.min(response_length).detach().item(),
        'response_length/hist': wandb.Histogram(response_length.detach()),
        # prompt length
        'prompt_length/mean': torch.mean(prompt_length).detach().item(),
        'prompt_length/max': torch.max(prompt_length).detach().item(),
        'prompt_length/min': torch.min(prompt_length).detach().item(),
    }

    if 'token_level_rewards' in batch.batch.keys():
        sequence_reward = batch.batch['token_level_rewards'].sum(-1)
        metrics.update({
            # reward
            'critic/rewards/mean': torch.mean(sequence_reward).detach().item(),
            'critic/rewards/max': torch.max(sequence_reward).detach().item(),
            'critic/rewards/min': torch.min(sequence_reward).detach().item(),
        })
    if 'values' in batch.batch.keys():
        values = batch.batch['values']
        metrics.update({
            # values
            'critic/values/mean': masked_mean(values, response_mask).detach().item(),
            'critic/values/max': torch.max(values[response_mask.bool()]).detach().item(),
            'critic/values/min': torch.min(values[response_mask.bool()]).detach().item(),
        })
    if 'returns' in batch.batch.keys():
        returns = batch.batch['returns']
        metrics.update({
            # returns
            'critic/returns/mean': masked_mean(returns, response_mask).detach().item(),
            'critic/returns/max': torch.max(returns[response_mask.bool()]).detach().item(),
            'critic/returns/min': torch.min(returns[response_mask.bool()]).detach().item(),
        })

    return metrics


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            # dwn: whether to fix kl coefficient
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        # dwn: check consistency for GRPO
        self.use_grpo = self.config.algorithm.use_grpo
        self.group_size = self.config.actor_rollout_ref.actor.group_size
        print(f'Use GRPO: {self.use_grpo}')
        if self.use_grpo:
            print(f'GRPO Group size: {self.group_size}')
            assert self.group_size > 1, f'Using GRPO, bad group size, got {self.group_size}'
            assert self.config.algorithm.adv_estimator in ['norm_os', 'norm_ps'], f'Using GRPO, bad adv_estimator, got {self.config.algorithm.adv_estimator}'
            assert self.config.actor_rollout_ref.actor.grpo_kl_coeff > 0.0, f'Using GRPO, bad grpo_kl_coeff, got {self.config.actor_rollout_ref.actor.grpo_kl_coeff}'
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size *= self.group_size
        else:
            assert self.group_size == 1, f'Not using GRPO, bad group size, got {self.group_size}'
            assert self.config.algorithm.adv_estimator == 'gae', f'Not using GRPO, bad adv_estimator, got {self.config.algorithm.adv_estimator}'
            assert self.config.actor_rollout_ref.actor.grpo_kl_coeff == 0.0, f'Not using GRPO, bad grpo_kl_coeff, got {self.config.actor_rollout_ref.actor.get("grpo_kl_coeff", 0.0)}'

        self._create_dataloader(use_grpo=self.use_grpo, group_size=self.group_size)

    def _create_dataloader(self, use_grpo=False, group_size=1):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn, get_grpo_collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         epochs=self.config.trainer.total_epochs)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=False,
                                           drop_last=True,
                                           collate_fn=get_grpo_collate_fn(group_size=group_size) if use_grpo else collate_fn) # dwn: during training, we modify collate_fn for GRPO

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader)

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        # dwn: for each resource pool, map role to its ray class
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'norm_os':
            # dwn: support GRPO (outcome-supervised)
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'norm_ps':
            # dwn: support GRPO (process-supervised)
            self.use_critic = False
            raise NotImplementedError
        else:
            # support ReMax
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        # dwn: for each resource pool, map role to its worker group
        all_wg = {}
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        global_steps = 0

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_training', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')

        for batch_dict in self.train_dataloader:
            print(f'[Step {global_steps}]')
            # dwn: check if v.shape[0] of all k,v is equal to train_batch_size * group_size
            for k, v in batch_dict.items():
                assert v.shape[0] == self.config.data.train_batch_size * self.group_size, f'Bad batch_size for {k}, expected {self.config.data.batch_size=} * {self.group_size=}, got {v.shape[0]}'
            metrics = {}

            batch: DataProto = DataProto.from_single_dict(batch_dict)
            # batch = batch.to('cuda')

            # pop those keys for generation
            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

            # generate a batch
            print('generation start')
            with Timer(name='gen', logger=None) as timer:
                # dwn:
                # prompt -> prompt + response

                # in: [input_ids, attention_mask, position_ids] -> out: [prompts, responses, input_ids, attention_mask, position_ids, old_log_probs, ...]

                # out.prompts = in.input_ids
                # out.responses = rollout's output token ids (not including input token ids)
                # out.input_ids = in.input_ids + out.responses
                # out.attention_mask = attention mask of out.input_ids
                # out.position_ids = position ids of out.input_ids
                # out.old_log_probs = log prob of out.responses

                # subsequently ref and critic need to compute per-token log prob corresponding to the response
                # now that input_ids has become prompt + response
                # to make that happen, we should use the forward method to obtain per-token log prob
                # in fact, the calculation of old_log_prob is also done via this method
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            metrics['timing/gen'] = timer.last
            print(f'generation end in {metrics["timing/gen"]:.2f} seconds')

            batch = batch.union(gen_batch_output)

            if self.use_reference_policy:
                # compute reference log_prob
                print('compute reference log_prob start')
                with Timer(name='ref', logger=None) as timer:
                    # dwn: "ref_log_prob"
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)
                metrics['timing/ref'] = timer.last
                print(f'compute reference log_prob end in {metrics["timing/ref"]:.2f} seconds')

            # dwn: only compute values when we use critic
            if self.use_critic:
                # compute values
                print('compute values start')
                with Timer(name='values', logger=None) as timer:
                    # dwn: "values", used for gae and clipping vpreds
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)
                metrics['timing/values'] = timer.last
                print(f'compute values end in {metrics["timing/values"]:.2f} seconds')

            print('compute advantages start')
            with Timer(name='adv', logger=None) as timer:
                # compute scores. Support both model and function-based.
                # We first compute the scores using reward model. Then, we call reward_fn to combine
                # the results from reward model and rule-based results.
                if self.use_rm:
                    # we first compute reward model score
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                # we combine with rule-based rm
                reward_tensor = self.reward_fn(batch)
                batch.batch['token_level_scores'] = reward_tensor

                # compute rewards. apply_kl_penalty if available
                # dwn: 
                # use ref_log_prob and old_log_prob to calculate kl penalty, and apply to rewards
                # critic/kl records the average of kl penalty
                # only apply kl penalty when we use critic
                if self.use_critic:
                    batch, kl_metrics = apply_kl_penalty(batch,
                                                        kl_ctrl=self.kl_ctrl,
                                                        kl_penalty=self.config.algorithm.kl_penalty)
                    metrics.update(kl_metrics)

                # compute advantages, executed on the driver process
                if self.use_grpo:
                    # dwn:
                    # if use GRPO, use reward normalization to calculate advantages
                    batch = compute_advantage(batch,
                                                adv_estimator=self.config.algorithm.adv_estimator,
                                                group_size=self.group_size)
                else:
                    # dwn: 
                    # if use PPO, use rewards and values to calculate gae
                    # returns = advantages + values
                    batch = compute_advantage(batch,
                                                self.config.algorithm.gamma,
                                                self.config.algorithm.lam,
                                                adv_estimator=self.config.algorithm.adv_estimator)
            metrics['timing/adv'] = timer.last
            print(f'compute advantages end in {metrics["timing/adv"]:.2f} seconds')

            # update critic
            if self.use_critic:
                print('update critic start')
                with Timer(name='update_critic', logger=None) as timer:
                    # dwn:
                    # critic loss = 0.5 * (values - returns) ** 2 = 0.5 * advantages ** 2
                    # in practic we calculate critic as follows:
                    # we use micro-batches to update the critic, during update, we use critic to calculate vpreds (the values produced by the partially updated critic)
                    # because the critic has changed, we expect vpreds to differ from values
                    # we use vpreds to calculate critic loss bacause values might to be too outdated
                    # to prevent critic from varying too much, we clip vpreds to be close to values
                    # critic loss = 0.5 * max( (vpreds - returns) ** 2, (vpredsclipped - returns) ** 2 )
                    critic_output = self.critic_wg.update_critic(batch)
                metrics['timing/update_critic'] = timer.last
                critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                metrics.update(critic_output_metrics)
                print(f'update critic end in {metrics["timing/update_critic"]:.2f} seconds')

            # implement critic warmup
            # dwn: don't update actor during warmup stage, because critic needs more updates
            if self.config.trainer.critic_warmup <= global_steps:
                # update actor
                print('update actor start')
                with Timer(name='update_actor', logger=None) as timer:
                    # dwn:
                    # we use micro-batches to update the actor, during update, we use actor to calculate log_probs (the log_probs produced by the partially updated actor)
                    # then we use importance sampling to calculate clipped objective
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                metrics['timing/update_actor'] = timer.last
                actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                metrics.update(actor_output_metrics)
                print(f'update actor end in {metrics["timing/update_actor"]:.2f} seconds')
            
            # validate
            if self.val_reward_fn is not None and (global_steps + 1) % self.config.trainer.test_freq == 0:
                print('validate start')
                with Timer(name='testing', logger=None) as timer:
                    val_metrics: dict = self._validate()
                    val_metrics = {f'val/{key}': val for key, val in val_metrics.items()}
                metrics['timing/testing'] = timer.last
                metrics.update(val_metrics)
                print(f'validate end in {metrics["timing/testing"]:.2f} seconds')

            # collect metrics
            data_metrics = compute_data_metrics(batch=batch)
            metrics.update(data_metrics)

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=global_steps)

            if self.config.trainer.save_freq > 0 and (global_steps + 1) % self.config.trainer.save_freq == 0:
                actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                                f'global_step_{global_steps}')
                actor_remote_path = os.path.join(self.config.trainer.default_hdfs_dir, 'actor')
                self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

                if self.use_critic:
                    critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                                        f'global_step_{global_steps}')
                    critic_remote_path = os.path.join(self.config.trainer.default_hdfs_dir, 'critic')
                    self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

            global_steps += 1

        # perform validation after training
        if self.val_reward_fn is not None:
            val_metrics = self._validate()
            pprint(f'Final validation metrics: {val_metrics}')
