# coding: utf-8

import argparse
import math
from copy import deepcopy

import numpy as np

from evaluator import Evaluator
from parking_entity.mycar import My_Car
from parking_env import ParkingEnv1, Car
from td3 import TD3
from util.map_load import *
from util.util import *

# from multiprocessing import Queue, Process

parking_group = load_dest_v2(3)


def get_a(observation):
    li = []
    index = 0
    for x in observation:
        if index == 3:
            li.append(math.cos(x))
            li.append(math.sin(x))
        else:
            li.append(x)
        # elif index < 5:
        #     li.append(deepcopy(x))

        index += 1
    return tuple(li)


def sigma_noise(action, sigma, beishu=1):
    normal = np.random.normal(0, 1, action.shape[0])
    noise_clip = 2 * sigma * beishu
    noise = normal * sigma * beishu
    noise = np.clip(noise, -noise_clip, noise_clip)
    action = action + noise
    action = np.clip(action, -1., 1.)
    return action


def train(num_iterations, agent, env, evaluate, validate_steps, output, epsilon_sigma, greedy_epsilon,
          max_episode_length=None,
          debug=False):
    global step_num_success_sum, observation
    global step_num_success_all
    global step_num_collide_sum
    global step_num_collide_all

    agent.is_training = True
    observation = None
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    bepisode = success = 0
    index = 0
    dv_sum = 0
    ddelta_sum = 0
    index_all = 0
    dv_sum_all = 0
    ddelta_sum_all = 0
    success_all = 0
    success_over_85_num = 0
    greedy = greedy_epsilon
    observation_all = None
    while step < num_iterations:
        # epsilon_sigma_temp = (num_iterations - step) * epsilon_sigma / num_iterations
        # agent.sigma = epsilon_sigma_temp
        # greedy = greedy_epsilon
        if bepisode > 0 and step % (num_iterations // 10000) == 0:
            print('Training {:.2f}%, success {:.2f}%'.format(
                step / num_iterations * 100, success / bepisode * 100))
            if step_num_collide_all != 0:
                print('撞车平均:' + "{:.2f}".format(step_num_collide_sum / step_num_collide_all) + '个step')
            if step_num_success_all != 0:
                print('成功停车:' + "{:.2f}".format(step_num_success_sum / step_num_success_all) + '个step')
            print('平均dv ' + str(dv_sum / index))
            print('平均ddelta ' + str(ddelta_sum / index))
            print('reset_x:{}, reset_y:{}, success_over_85_num: {}'
                  .format(env.parking_group.reset_x, env.parking_group.reset_y, success_over_85_num))
            temp = 0.25
            if step_num_success_sum == 0:
                temp = 1
            if greedy == 1:
                temp = 0.25
            greedy = temp
            if (step / num_iterations * 10000) % 75 == 0:
                # env.dec_car_reset_range()
                pass
            if success / bepisode > 0.85:
                success_over_85_num += 1
            else:
                success_over_85_num = 0
            if success_over_85_num >= 3:
                success_over_85_num = 0
                env.add_car_reset_range()
            dv_sum = 0
            ddelta_sum = 0
            index = 0
            step_num_success_sum = 0
            step_num_success_all = 0
            step_num_collide_sum = 0
            step_num_collide_all = 0
            print('---------------------------------------------')
            bepisode = success = 0
            agent.save_model(output)

        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            a = get_a(observation)
            # torch.cat((to_tensor(np.array([a])), out)
            agent.reset(a)

        # agent pick action ...
        if step <= args.warmup or np.random.uniform(0, 1) < greedy:
            action = agent.random_action()
        else:
            a = get_a(observation)
            action = agent.select_action(a)
            action = sigma_noise(action, epsilon_sigma)
            # epsilon = np.random.normal(0, epsilon_sigma, action.shape[0])
            # action = np.clip(action + epsilon, -1, 1)  # random noise
            # TODO: Normalization

            dv, ddelta = action
            dv_sum_all += dv
            ddelta_sum_all += ddelta
            index_all += 1

        dv, ddelta = action
        dv_sum += dv
        ddelta_sum += ddelta
        index += 1

        # parking_env response with next_observation, reward, terminate_info
        observation2, reward, done, info, step_num = env.step(action)
        if info == 'in_slot':
            success += 1
            step_num_success_sum += step_num
            step_num_success_all += 1
            if step > args.warmup:
                success_all += 1

            # print('成功停车:' + str(step_num) + '轮')

        if info == 'collide':
            step_num_collide_sum += step_num
            step_num_collide_all += 1
            # print('撞车:' + str(step_num) + '轮')

        # if reward < -10000:
        #     print('reward: ', reward)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True
            # 碰撞位置离停车位越远，惩罚越大
            # reward -= env.position_penalty * env.parking_group.dis_car_target(env.car) + env.collision_penalty

            # 如果汽车的一部分进入了停车位
            # intersect_area = env.parking_group.get_intersect_area(env.car)

            # 奖励，如果汽车进入车位的部分越多，奖励越大
            # reward += 0.6 * env.in_slot_reward * intersect_area / (env.car.car_width * env.car.car_height)

        # agent observe and update policy
        a = get_a(observation2)
        agent.observe(reward, a, done)  # save observation to memory

        # if step > args.warmup:
        if step > args.warmup:
            agent.update_policy()

        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            def policy(x):
                return agent.select_action(x, decay_epsilon=False)

            validate_reward = evaluate(
                env, policy, debug=False, visualize=False)
            if debug:
                prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(
                    step, validate_reward))

        # [optional] save intermideate model
        if step % int(num_iterations / 3) == 0:
            agent.save_model('saved_models')

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            # print(info, reward)
            if debug:
                prGreen('#{}: episode_reward:{} steps:{}'.format(
                    episode, episode_reward, step))

            # a = get_a(observation)
            # agent.memory.append(
            #     deepcopy(a),
            #     agent.select_action(deepcopy(a)),
            #     0., False
            # )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            bepisode += 1
    return dv_sum_all / index_all, ddelta_sum_all / index_all, success_all


def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()

    def policy(x):
        return agent.select_action(x, decay_epsilon=False)

    if evaluate is not None:
        # 20
        for i in range(num_episodes):
            validate_reward = evaluate(
                env, policy, debug=debug, visualize=visualize, save=True)
            print(validate_reward)
            if debug:
                prYellow('[Evaluate] #{}: mean_reward:{}'.format(
                    i, validate_reward))


observation = None
observation2 = None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str,
                        help='support option: train/test')
    parser.add_argument('--pretrained', default=1,
                        type=int, help='use pretrained model')
    parser.add_argument('--env', default='Pendulum-v1',
                        type=str, help='open-ai gym environment')

    # 400
    parser.add_argument('--hidden1', default=400, type=int,
                        help='hidden num of first fully connect layer')
    # 300
    parser.add_argument('--hidden2', default=300, type=int,
                        help='hidden num of second fully connect layer')

    # 0.001  0.0001
    parser.add_argument('--rate', default=0.001,
                        type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float,
                        help='policy net learning rate (only for DDPG)')

    # 10000
    parser.add_argument('--warmup', default=128, type=int,
                        help='time without training but only filling the replay memory')

    parser.add_argument('--update_frequency', default=2, type=int,
                        help='how many steps to update the parameter of actor')

    # 0.99
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=128,
                        type=int, help='minibatch size')
    # default=6000000
    parser.add_argument('--rmsize', default=6000000,
                        type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')

    # 0.005
    parser.add_argument('--tau', default=0.001, type=float,
                        help='moving average for target network')

    # 0.3
    parser.add_argument('--epsilon_sigma', default=0.15, type=float,
                        help='standard deviation of actor\'s epsilon noise')
    parser.add_argument('--greedy_epsilon', default=0.25,
                        type=float, help='epsilon greedy')
    parser.add_argument('--ou_theta', default=0.15,
                        type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.3,
                        type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=20, type=int,
                        help='how many episode to perform during validate experiment')

    # 500
    parser.add_argument('--max_episode_length', default=600, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int,
                        help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')

    # vis
    parser.add_argument('--visualize', default=1, type=int,
                        help='visualize parking process')
    parser.add_argument('--init_w', default=0.0001, type=float, help='')
    # 200000
    parser.add_argument('--train_iter', default=20000000,
                        type=int, help='train iters each timestep')
    # 50000
    parser.add_argument('--epsilon', default=50000, type=int,
                        help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default',
                        type=str, help='Resuming model path for testing')
    parser.add_argument('--max_img_size', default='1000',
                        type=int, help='max image size of image loader')
    parser.add_argument('--total_dest_size', default='2',
                        type=int, help='the number of json file in dest dir')
    parser.add_argument('--is_priority', default='0',
                        type=int, help='是否采用优先经验回放')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'saved_models'

    # img_loader = LRU_Image_loader.get_instance(args.max_img_size)

    car = Car(3, 1, 1, 2)

    '''
    def __init__(self, car, l_slot, w_slot, max_angle, mode, position_penalty,
             final_angle_penalty, in_slot_reward=100000, collision_penalty=100000
             , angle_change_penalty=10, speed_penalty=10,
             reverse_penalty=100, dt=0.02, max_dv=1., max_ddelta=0.2, max_speed=2.,
             xbound=15, ybound=15, total_dest_size=1,
             close_to_reward=1000, visualize=True, **init_config):
             4000
    '''
    # 增加rank
    # TODO: 碰撞代码，水平停车，左侧撞不到(左上角的停车位)
    # q_ = Queue()

    list = []
    decount = 1
    collision_penalty_list = [10000 / decount] * 10
    in_slot_reward = 10000 / decount
    reset_x = 0.6
    reset_y = 0.52
    for collision_penalty in collision_penalty_list:
        car_1 = My_Car()
        env = ParkingEnv1(car_1, 6, 4, np.pi / 4, 'reverse',
                          1 / decount, 5 / decount, in_slot_reward, collision_penalty,
                          10 / decount, 10 / decount, 50 / decount, dt=0.1,
                          visualize=args.visualize, parking_group=None)  # visualize=args.mode=='test'
        env.set_car_reset_range(reset_x, reset_y)
        # p1 = Process(target=cal_reward, args=(q_, ))
        # p1.start()
        if args.seed > 0:
            np.random.seed(args.seed)

        dim_states = env.dim_states
        dim_actions = env.dim_actions

        agent = TD3(dim_states, dim_actions, args)

        evaluate = Evaluator(args.validate_episodes,
                             args.validate_steps, args.output, max_episode_length=args.max_episode_length)

        if args.mode == 'train':
            step_num_success_sum = 0
            step_num_collide_sum = 0
            step_num_success_all = 0
            step_num_collide_all = 0
            dv, ddelta, success_all = train(args.train_iter, agent, env, None, args.validate_steps, args.resume,
                                            args.epsilon_sigma,
                                            args.greedy_epsilon, max_episode_length=args.max_episode_length,
                                            debug=args.debug)
            print('-------------一轮训练结束， 总结----------------')
            print('\n\n\n\n\n\n')
            print(collision_penalty, 'dv: {:.3f}'.format(dv), 'ddelta: {:.3f}'.format(ddelta), success_all)
            list.append([collision_penalty, 'dv: {:.3f}'.format(dv), 'ddelta: {:.3f}'.format(ddelta), success_all])
            print('\n\n\n\n\n\n')
            print('-------------一轮训练结束， 总结----------------')

        elif args.mode == 'test':
            test(args.validate_episodes, agent, env, evaluate, args.resume,
                 visualize=False, debug=args.debug)

        # elif args.mode == 'operator':
        #     operator()

        else:
            raise RuntimeError('undefined mode {}'.format(args.mode))
        reset_x, reset_y = env.get_car_reset_range()

    for item in list:
        print(item)
