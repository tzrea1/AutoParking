import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from memory_priority import PriReplayBuffer
from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *

# from ipdb import set_trace as debug
from util.util import hard_update, USE_CUDA, to_tensor, soft_update, to_numpy

criterion = nn.MSELoss()


class TD3(object):
    def __init__(self, dim_states, dim_actions, args):

        if args.seed > 0:
            self.seed(args.seed)

        self.dim_states = dim_states
        self.dim_actions = dim_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w,
        }

        self.actor = Actor(self.dim_states, self.dim_actions, **net_cfg)
        self.actor_target = Actor(self.dim_states, self.dim_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        self.critic1 = Critic(self.dim_states, self.dim_actions, **net_cfg)
        self.critic1_target = Critic(
            self.dim_states, self.dim_actions, **net_cfg)
        self.critic2 = Critic(self.dim_states, self.dim_actions, **net_cfg)
        self.critic2_target = Critic(
            self.dim_states, self.dim_actions, **net_cfg)

        if args.pretrained:
            self.load_weights(args.resume)

        self.critic1_optim = Adam(self.critic1.parameters(), lr=args.rate)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=args.rate)
        self.is_priority = args.is_priority

        # Make sure target is with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic1_target, self.critic1)
        hard_update(self.critic2_target, self.critic2)

        # Create replay buffer
        if not self.is_priority:
            self.memory = SequentialMemory(
                limit=args.rmsize, window_length=args.window_length)
        else:
            self.priority_memory = PriReplayBuffer(size=args.rmsize)

        self.random_process = OrnsteinUhlenbeckProcess(
            size=dim_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon
        self.sigma = args.epsilon_sigma
        self.update_frequency = args.update_frequency

        #
        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True
        self.step = 0

        #
        if USE_CUDA:
            self.cuda()

    def sigma_noise(self, action, beishu):
        normal = np.random.normal(0, 1, (self.batch_size, self.dim_actions))
        noise_clip = 2 * self.sigma * beishu
        noise = normal * self.sigma * beishu
        noise = np.clip(noise, -noise_clip, noise_clip)
        action = to_numpy(action)
        action = action + noise
        action = np.clip(action, -1., 1.)
        return to_tensor(action)

    def update_policy(self):
        """Fetch an experience batch with size of `batch_size` """
        b_idx = None
        ISWeights = 1
        if not self.is_priority:
            # Sample batch
            state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch = self.memory.sample_and_split(
                self.batch_size)
        else:
            b_idx, state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch, ISWeights = self.priority_memory.sample(
                self.batch_size)
            ISWeights = to_tensor(ISWeights).cuda()

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = torch.min(self.critic1_target([
                to_tensor(next_state_batch),
                self.sigma_noise(self.actor_target(to_tensor(next_state_batch)), 2),
            ]), self.critic2_target([
                to_tensor(next_state_batch),
                self.sigma_noise(self.actor_target(to_tensor(next_state_batch)), 2),
            ]))
            # TODO: Check dim

        target_q_batch = to_tensor(reward_batch) + \
                         self.discount * to_tensor(terminal_batch.astype(float)) * next_q_values
        target_q_batch = target_q_batch

        # Critic update
        self.critic1.zero_grad()
        self.critic2.zero_grad()

        q1_batch = self.critic1(
            [to_tensor(state_batch), to_tensor(action_batch)])
        q2_batch = self.critic2(
            [to_tensor(state_batch), to_tensor(action_batch)])

        value_loss_1 = torch.mul((q1_batch - target_q_batch) ** 2, ISWeights).mean()
        value_loss_1.backward()
        self.critic1_optim.step()

        value_loss_2 = torch.mul((q2_batch - target_q_batch) ** 2, ISWeights).mean()
        value_loss_2.backward()
        self.critic2_optim.step()

        if self.is_priority:
            abs_err = torch.abs(q2_batch - target_q_batch).flatten().cpu().detach().numpy()
            self.priority_memory.batch_update(b_idx, abs_err)  # update priority

        self.step += 1

        if self.step % self.update_frequency == 0:
            # Actor update
            self.actor.zero_grad()

            policy_loss = -self.critic1([
                to_tensor(state_batch),
                self.actor(to_tensor(state_batch))
            ])

            policy_loss = policy_loss.mean()

            # print('pocily_loss: ', policy_loss)

            policy_loss.backward()
            self.actor_optim.step()

            # Target update
            if self.step % (self.update_frequency * 3) == 0:
                soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic1_target, self.critic1, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic1.eval()
        self.critic1_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic1.cuda()
        self.critic1_target.cuda()
        self.critic2.cuda()
        self.critic2_target.cuda()

    def observe(self, r_t, s_t1, done):
        """Append observation to memory"""
        if self.is_training:
            if not self.is_priority:
                self.memory.append(self.s_t, self.a_t, r_t, done, s_t1)
            else:
                self.priority_memory.store((self.s_t, self.a_t, r_t, done, s_t1))
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1., 1., self.dim_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        action += self.is_training * \
                  max(self.epsilon, 0) * self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None:
            return

        self.actor.load_state_dict(
            torch.load('{}/actor.pth'.format(output))
        )

        self.critic1.load_state_dict(
            torch.load('{}/critic1.pth'.format(output))
        )

        self.critic2.load_state_dict(
            torch.load('{}/critic2.pth'.format(output))
        )

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pth'.format(output)
        )
        torch.save(
            self.critic1.state_dict(),
            '{}/critic1.pth'.format(output)
        )
        torch.save(
            self.critic2.state_dict(),
            '{}/critic2.pth'.format(output)
        )

    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
