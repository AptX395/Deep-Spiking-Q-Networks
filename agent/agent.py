# coding = utf-8

import logging
import random
import numpy
import torch

logger = logging.getLogger("file_logger")


class DqnAgent:
    def __init__(self, eval_env, main_net, writer, train_env=None, target_net=None, replay_memory=None, spike_monitor=None):
        self._eval_env = eval_env
        self._main_net = main_net
        self._writer = writer
        self._train_env = train_env
        self._target_net = target_net
        self._replay_memory = replay_memory
        self._spike_monitor = spike_monitor
        self._device = self._main_net.device

    def learn(self, timestep_num, replay_start_size, minibatch_size, discount_factor, update_freq, target_net_update_freq,
              eval_freq, eval_episode_num, eval_epsilon, init_epsilon, final_epsilon, final_epsilon_frame, model_path):
        logger.info("Start learning")
        self._main_net.train()
        self._update_target_net()
        self._target_net.eval()
        self._explore(replay_start_size, discount_factor)
        timestep = 0
        max_average_score = numpy.NINF

        # Training loop.
        while timestep < timestep_num:
            is_done = False
            next_frame_stack = self._train_env.reset()  # Reset the Atari environment and the frame stack.

            # The episode loop of the Atari environment.
            while not is_done:
                frame_stack = next_frame_stack
                action = self._select_action(timestep, init_epsilon, final_epsilon, final_epsilon_frame, frame_stack)
                next_frame_stack, reward, is_done, _ = self._train_env.step(action)

                # The 'mask' was calculated in advance to simplify the updating process of the neural network.
                mask = 0.0 if is_done else discount_factor

                self._replay_memory.append(transition=(frame_stack, action, reward, mask, next_frame_stack))

                # Update the main network by minibatch SGD.
                if timestep % update_freq == 0:
                    minibatch = self._replay_memory.sample(minibatch_size)
                    loss = self._update_main_net(minibatch[0], minibatch[1], minibatch[2], minibatch[3], minibatch[4])
                    self._writer.add_scalar(tag="learn/loss", scalar_value=loss, global_step=timestep)

                # Update the target network by copying the parameters of the main network.
                if timestep % target_net_update_freq == 0:
                    self._update_target_net()

                # Evaluate the performence of the main network on the Atari environment(for evaluation).
                if timestep % eval_freq == 0:
                    max_average_score = self._evaluate(eval_episode_num, eval_epsilon, timestep, max_average_score, model_path)

                timestep += 1

        logger.info("Learning is done")

    def play(self, model_path, eval_episode_num, eval_epsilon):
        logger.info("Start playing")
        self._load_model(model_path)
        self._main_net.eval()
        scores = list()
        max_qs = list()
        max_q_stds = list()

        # Evaluation loop.
        for episode in range(eval_episode_num):
            is_done = False
            frame_stack = self._eval_env.reset()  # Reset the Atari environment and the frame stack.
            score = 0.0
            max_q = numpy.NINF
            max_q_std = numpy.NINF

            # The episode loop of the Atari environment.
            while not is_done:
                action, max_q, max_q_std = self._get_action(eval_epsilon, frame_stack, max_q, max_q_std)
                frame_stack, reward, is_done, _ = self._eval_env.step(action)
                score += reward

            scores.append(score)
            max_qs.append(max_q)
            max_q_stds.append(max_q_std)
            self._writer.add_scalar(tag="play/score", scalar_value=score, global_step=episode)
            self._writer.add_scalar(tag="play/max_q", scalar_value=max_q, global_step=episode)
            self._writer.add_scalar(tag="play/max_q_std", scalar_value=max_q_std, global_step=episode)

        average_score = numpy.mean(scores)
        max_score = numpy.max(scores)
        score_std = numpy.std(scores)
        average_max_q = numpy.mean(max_qs)
        average_max_q_std = numpy.mean(max_q_stds)
        logger.info(f"Playing is done.\naverage_score: {average_score}\tmax_score: {max_score}\tscore_std: {score_std}\t"
                    f"average_max_q: {average_max_q}\taverage_max_q_std: {average_max_q_std}")

    # -------------------------------------------------------------------------------------------------------------------------

    def _update_target_net(self):
        self._target_net.load_state_dict(state_dict=self._main_net.state_dict())

    def _explore(self, explore_timestep, discount_factor):
        logger.info("Start exploring")
        timestep = 0

        # Exploring loop.
        while timestep < explore_timestep:
            is_done = False
            next_frame_stack = self._train_env.reset()  # Reset the Atari environment and the frame stack.

            # The episode loop of the Atari environment.
            while not is_done:
                frame_stack = next_frame_stack
                action = self._train_env.action_space.sample()
                next_frame_stack, reward, is_done, _ = self._train_env.step(action)

                # The 'mask' was calculated in advance to simplify the updating process of the neural network.
                mask = 0.0 if is_done else discount_factor

                self._replay_memory.append(transition=(frame_stack, action, reward, mask, next_frame_stack))
                timestep += 1

        logger.info("Exploring is done")

    def _select_action(self, timestep, init_epsilon, final_epsilon, final_epsilon_frame, frame_stack):
        # Select an action at random with probability = `epsilon`, otherwise select the action with the highest Q value.
        if random.random() < self._epsilon(timestep, init_epsilon, final_epsilon, final_epsilon_frame):
            action = self._train_env.action_space.sample()
        else:
            state = self._frame_stack_to_state(frame_stack)
            output = self._main_net.predict(state)
            action = output.argmax().item()

        return action

    def _epsilon(self, timestep, init_epsilon, final_epsilon, final_epsilon_frame):
        # `epsilon` is linearly decayed within a set time step, and then the `epsilon` is fixed.
        if timestep < final_epsilon_frame:
            epsilon = (final_epsilon - init_epsilon) / final_epsilon_frame * timestep + init_epsilon
        else:
            epsilon = final_epsilon

        return epsilon

    def _frame_stack_to_state(self, frame_stack, is_batch=False):
        # To save memory, the frames stored in the experience replay memory are integer, therefore need to be normalized.
        state = numpy.array(frame_stack, dtype=numpy.float32) / 255.0

        # From numpy shape = (batch, height, width, channel) to pytorch shape (batch, channel, height, width).
        if is_batch:
            state = numpy.transpose(state, axes=(0, 3, 1, 2))
            state = torch.tensor(state, dtype=torch.float32, device=self._device)
        else:
            state = numpy.transpose(state, axes=(2, 0, 1))
            state = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(dim=0)

        return state

    def _update_main_net(self, frame_stacks, actions, rewards, masks, next_frame_stacks):
        y = self._calculate_y(rewards, masks, next_frame_stacks)
        q = self._calculate_q(frame_stacks, actions)
        loss = self._main_net.update(q, y)
        return loss

    def _calculate_y(self, rewards, masks, next_frame_stacks):
        next_states = self._frame_stack_to_state(next_frame_stacks, is_batch=True)
        output = self._target_net.predict(next_states)
        max_target_q = output.max(dim=1)[0].unsqueeze(dim=-1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self._device).unsqueeze(dim=-1)
        masks = torch.tensor(masks, dtype=torch.float32, device=self._device).unsqueeze(dim=-1)
        y = rewards + masks * max_target_q
        return y

    def _calculate_q(self, frame_stacks, actions):
        states = self._frame_stack_to_state(frame_stacks, is_batch=True)
        output = self._main_net.forward(states)
        actions = torch.tensor(actions, dtype=torch.int64, device=self._device).unsqueeze(dim=-1)
        q = output.gather(dim=1, index=actions)
        return q

    def _evaluate(self, eval_episode_num, eval_epsilon, timestep, max_average_score, model_path):
        scores = list()
        max_qs = list()

        # Evaluation loop.
        for episode in range(eval_episode_num):
            is_done = False
            frame_stack = self._eval_env.reset()  # Reset the Atari environment and the frame stack.
            score = 0.0
            max_q = numpy.NINF
            max_q_std = numpy.NINF  # This is useless when evaluating.

            # The episode loop of the Atari environment.
            while not is_done:
                action, max_q, max_q_std = self._get_action(eval_epsilon, frame_stack, max_q, max_q_std)
                frame_stack, reward, is_done, _ = self._eval_env.step(action)
                score += reward

            scores.append(score)
            max_qs.append(max_q)

        average_score = numpy.mean(scores)
        score_std = numpy.std(scores)
        average_max_q = numpy.mean(max_qs)
        max_average_score = self._record_data(average_score, score_std, average_max_q, timestep, max_average_score, model_path)
        return max_average_score

    def _record_data(self, average_score, score_std, average_max_q, timestep, max_average_score, model_path):
        self._writer.add_scalar(tag="evaluate/average_score", scalar_value=average_score, global_step=timestep)
        self._writer.add_scalar(tag="evaluate/score_std", scalar_value=score_std, global_step=timestep)
        self._writer.add_scalar(tag="evaluate/average_max_q", scalar_value=average_max_q, global_step=timestep)
        logger.info(f"timestep: {timestep}\taverage_score: {average_score}\tscore_std: {score_std}\t"
                    f"average_max_q: {average_max_q}")

        if max_average_score <= average_score:
            max_average_score = average_score
            self._writer.add_scalar(tag="evaluate/max_average_score", scalar_value=max_average_score, global_step=timestep)
            logger.info(f"timestep: {timestep}\tmax_average_score: {max_average_score}")
            self._save_model(model_path)

        return max_average_score

    def _get_action(self, eval_epsilon, frame_stack, max_q, max_q_std):
        # Select an action at random with probability = `eval_epsilon`, otherwise select the action with the highest Q value.
        if random.random() < eval_epsilon:
            action = self._eval_env.action_space.sample()
        else:
            state = self._frame_stack_to_state(frame_stack, is_batch=False)
            output = self._main_net.predict(state)
            action = output.argmax().item()
            q = output[0][action].item()

            if max_q < q:
                max_q = q
                max_q_std = output.std(dim=1).item()

        return action, max_q, max_q_std

    def _save_model(self, model_path):
        torch.save(self._main_net.state_dict(), model_path)

    def _load_model(self, model_path):
        self._main_net.load_state_dict(state_dict=torch.load(model_path))


class DsqnAgent(DqnAgent):
    def play(self, model_path, eval_episode_num, eval_epsilon):
        logger.info("Start playing")
        self._load_model(model_path)
        self._main_net.eval()
        self._spike_monitor.enable()
        scores = list()
        max_qs = list()
        max_q_stds = list()
        episode_avg_firing_rates = list()
        total_action_num = 0
        total_spike_num = 0

        # Evaluation loop.
        for episode in range(eval_episode_num):
            is_done = False
            frame_stack = self._eval_env.reset()  # Reset the Atari environment and the frame stack.
            score = 0.0
            max_q = numpy.NINF
            max_q_std = numpy.NINF
            avg_firing_rates = list()

            # The episode loop of the Atari environment.
            while not is_done:
                self._spike_monitor.reset()
                action, max_q, max_q_std = self._get_action(eval_epsilon, frame_stack, max_q, max_q_std)
                frame_stack, reward, is_done, _ = self._eval_env.step(action)
                score += reward

                if self._spike_monitor.module_dict["_conv1"].cnt:
                    avg_firing_rate = self._spike_monitor.get_avg_firing_rate().item()
                    avg_firing_rates.append(avg_firing_rate)
                    total_action_num += 1

                    for module in self._spike_monitor.module_dict.values():
                        total_spike_num += module.firing_time.item()

            scores.append(score)
            max_qs.append(max_q)
            max_q_stds.append(max_q_std)
            episode_avg_firing_rate = numpy.mean(avg_firing_rates)
            episode_avg_firing_rates.append(episode_avg_firing_rate)
            self._writer.add_scalar(tag="play/score", scalar_value=score, global_step=episode)
            self._writer.add_scalar(tag="play/max_q", scalar_value=max_q, global_step=episode)
            self._writer.add_scalar(tag="play/max_q_std", scalar_value=max_q_std, global_step=episode)
            self._writer.add_scalar(tag="play/episode_avg_firing_rate", scalar_value=episode_avg_firing_rate,
                                    global_step=episode)

        average_score = numpy.mean(scores)
        max_score = numpy.max(scores)
        score_std = numpy.std(scores)
        average_max_q = numpy.mean(max_qs)
        average_max_q_std = numpy.mean(max_q_stds)
        average_firing_rate = numpy.mean(episode_avg_firing_rates)
        spike_num_per_action = total_spike_num / total_action_num
        logger.info(f"Playing is done.\naverage_score: {average_score}\tmax_score: {max_score}\tscore_std: {score_std}\t"
                    f"average_max_q: {average_max_q}\taverage_max_q_std: {average_max_q_std}\t"
                    f"average_firing_rate: {average_firing_rate}\tspike_num_per_action: {spike_num_per_action}")
        self._spike_monitor.disable()

    def _get_action(self, eval_epsilon, frame_stack, max_q, max_q_std):
        # Select an action at random with probability = `eval_epsilon`, otherwise select the action with the highest Q value.
        if random.random() < eval_epsilon:
            action = self._eval_env.action_space.sample()
        else:
            state = self._frame_stack_to_state(frame_stack, is_batch=False)
            output = self._main_net.predict(state)
            action = output.argmax().item()
            q = output[0][action].item()

            if max_q < q:
                max_q = q
                max_q_std = output.std(dim=1).item()

        return action, max_q, max_q_std

    def _evaluate(self, eval_episode_num, eval_epsilon, timestep, max_average_score, model_path):
        scores = list()
        max_qs = list()

        # Evaluation loop.
        for episode in range(eval_episode_num):
            is_done = False
            frame_stack = self._eval_env.reset()  # Reset the Atari environment and the frame stack.
            score = 0.0
            max_q = numpy.NINF

            # The episode loop of the Atari environment.
            while not is_done:
                action, max_q, max_q_std = self._get_action(eval_epsilon, frame_stack, max_q, max_q_std)
                frame_stack, reward, is_done, _ = self._eval_env.step(action)
                score += reward

            scores.append(score)
            max_qs.append(max_q)

        average_score = numpy.mean(scores)
        score_std = numpy.std(scores)
        average_max_q = numpy.mean(max_qs)
        max_average_score = self._record_data(average_score, score_std, average_max_q, timestep, max_average_score, model_path)
        return max_average_score

    def _record_data(self, average_score, score_std, average_max_q, timestep, max_average_score, model_path):
        self._writer.add_scalar(tag="evaluate/average_score", scalar_value=average_score, global_step=timestep)
        self._writer.add_scalar(tag="evaluate/score_std", scalar_value=score_std, global_step=timestep)
        self._writer.add_scalar(tag="evaluate/average_max_q", scalar_value=average_max_q, global_step=timestep)
        logger.info(f"timestep: {timestep}\taverage_score: {average_score}\tscore_std: {score_std}\t"
                    f"average_max_q: {average_max_q}")
        index = torch.tensor(data=[0], device=self._device)

        if max_average_score <= average_score:
            max_average_score = average_score
            self._writer.add_scalar(tag="evaluate/max_average_score", scalar_value=max_average_score, global_step=timestep)
            logger.info(f"timestep: {timestep}\tmax_average_score: {max_average_score}")
            self._save_model(model_path)

        return max_average_score
