import datetime
import os
import copy
import json
import numpy as np
from pytz import timezone

from gamified_squad import GamifiedSquad
from agent import CustomAgent
import generic
import evaluate
SAVE_CHECKPOINT = 100000


def train():

    time_1 = datetime.datetime.now()
    config = generic.load_config()
    env = GamifiedSquad(config)
    env.split_reset("train")
    agent = CustomAgent(config, env.has_token_set)


    if config["general"]["visdom"]:
        # visdom
        import visdom
        viz = visdom.Visdom()
        plt_win = None
        eval_plt_win = None
        plt_q_value_win = None
        plt_steps_win = None
        eval_plt_steps_win = None
        viz_avg_ig_acc, viz_avg_qa_acc = [], []
        viz_avg_ig_q_value = []
        viz_eval_ig_acc, viz_eval_qa_acc, viz_eval_steps = [], [], []
        viz_avg_steps = []

    step_in_total = 0
    batch_no = 0
    episode_no = 0
    running_avg_qa_acc = generic.HistoryScoreCache(capacity=50)
    running_avg_ig_acc = generic.HistoryScoreCache(capacity=50)
    running_avg_qa_loss = generic.HistoryScoreCache(capacity=50)
    running_avg_ig_loss = generic.HistoryScoreCache(capacity=50)
    running_avg_ig_q_value = generic.HistoryScoreCache(capacity=50)
    running_avg_steps = generic.HistoryScoreCache(capacity=50)

    output_dir = "."
    data_dir = "."
    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_qa_acc_so_far = 0.0
    prev_performance = 0.0
    i_am_patient = 0
    # load model from checkpoint
    
    if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
        print("checkpoint already exist.")
        exit(0)

    if os.path.exists(data_dir + "/" + agent.load_graph_generation_model_from_tag + ".pt"):
        agent.load_pretrained_graph_generation_model(data_dir + "/" + agent.load_graph_generation_model_from_tag + ".pt")
    if agent.load_pretrained:
        if os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"):
            agent.load_pretrained_model(data_dir + "/" + agent.load_from_tag + ".pt")  # load partial graph
            agent.update_target_net()

    while(True):
        if episode_no > agent.max_episode:
            break
        np.random.seed(episode_no)
        env.seed(episode_no)
        obs, infos = env.reset()
        batch_size = len(obs)
        report = agent.report_frequency > 0 and (episode_no % agent.report_frequency <= max(episode_no - batch_size, 0) % agent.report_frequency)
        __save__ = episode_no % SAVE_CHECKPOINT <= max(episode_no - batch_size, 0) % SAVE_CHECKPOINT
        if report:    
            print("====================================================================================", episode_no)
            print("-- Q: %s" % (agent.bert_tokenizer.decode(infos[0]["q"]).encode('utf-8')))
            print("-- A: %s" % (infos[0]["a_string"][0].encode('utf-8')))

        agent.train()
        agent.init(obs, infos)
        quest_list = agent.get_game_quest_info(infos)
        agent.kg.push_batch_question(quest_list, [item["q_srl"] for item in infos])
        if agent.use_gt_graph:
            for i in range(batch_size):
                full_obs = infos[i]["full_obs"]
                full_srl = infos[i]["full_srl"]
                #####################################################
                for j in range(len(full_obs)):
                    agent.kg.push_one(i, full_obs[j], full_srl[j])

        previous_dynamics = None
        previous_belief = None
        input_quest, input_quest_mask, quest_id_list = agent.get_agent_inputs(quest_list)
        tmp_replay_buffer = []
        print_cmds = []
        prev_commands = ["restart" for _ in range(batch_size)]
        belief_buffer = []

        act_randomly = False if agent.noisy_net else episode_no < agent.learn_start_from_this_episode
        for _ in range(agent.max_nb_steps_per_episode):
            # generate commands
            if agent.noisy_net:
                agent.reset_noise()  # Draw a new set of noisy weights
            commands, replay_info, current_dynamics, current_belief = agent.act(obs, infos, input_quest, input_quest_mask, quest_id_list, prev_commands, previous_dynamics, previous_belief, random=act_randomly)

            tmp_replay_buffer.append(replay_info)
            obs, infos = env.step(commands)
            prev_commands = commands
            previous_dynamics = current_dynamics
            previous_belief = current_belief
            belief_buffer.append(current_belief)

            if agent.noisy_net and step_in_total % agent.update_per_k_game_steps == 0:
                agent.reset_noise()  # Draw a new set of noisy weights

            if episode_no >= agent.learn_start_from_this_episode and step_in_total % agent.update_per_k_game_steps == 0:
                interaction_loss, interaction_q_value = agent.update_interaction()
                if interaction_loss is not None:
                    running_avg_ig_loss.push(interaction_loss)
                    running_avg_ig_q_value.push(interaction_q_value)
                qa_loss = agent.update_qa()
                if qa_loss is not None:
                    running_avg_qa_loss.push(qa_loss)

            step_in_total += 1
            still_running = generic.to_np(replay_info[-1])
            print_cmds.append(commands[0] if still_running[0] else "--")
            if np.sum(still_running) == 0:
                break

        if report: 
            print(" / ".join(print_cmds).encode('utf-8'))
        # The agent has exhausted all steps, now answer question.
        chosen_head_tails = agent.answer_question_act(agent.naozi.get(), quest_list, current_belief)  # batch
        chosen_head_tails_np = generic.to_np(chosen_head_tails)
        chosen_answer_strings = generic.get_answer_strings(agent.naozi.get(), chosen_head_tails_np, agent.bert_tokenizer, agent.special_token_ids)
        answer_strings = [item["a_string"] for item in infos]
        answer_token_ids = [item["a"] for item in infos]

        qa_reward_np = generic.get_qa_reward(chosen_answer_strings, answer_strings)
        obs_strings = [agent.bert_tokenizer.decode(agent.naozi.get(i)) for i in range(batch_size)]
        ig_reward_np = generic.get_sufficient_info_reward(agent.naozi.get(), answer_token_ids)
        ig_reward = generic.to_pt(ig_reward_np, enable_cuda=False, type='float')  # batch

        # push qa experience into qa replay buffer
        replay_node_vocab = agent.kg.get_node_vocabulary()
        replay_relation_vocab = agent.kg.get_relation_vocabulary()
        replay_triplets = agent.kg.get_triplets()
        for b in range(batch_size):  # data points in batch
            is_prior = qa_reward_np[b] > agent.qa_reward_prior_threshold * agent.qa_replay_memory.avg_rewards()
            # if the agent is not in the correct state, do not push it into replay buffer
            if np.mean(ig_reward_np[b]) == 0.0:
                continue
            agent.qa_replay_memory.push(is_prior, qa_reward_np[b], agent.naozi.get_sentence_lists(b), quest_list[b], replay_node_vocab[b], replay_relation_vocab[b], replay_triplets[b], answer_token_ids[b], belief_buffer[-1][b].cpu() if belief_buffer[-1][b] is not None else None)

        # small positive reward whenever it answers question correctly
        masks_np = [generic.to_np(item[-1]) for item in tmp_replay_buffer]
        command_rewards_np = []
        for i in range(len(tmp_replay_buffer)):
            if i == len(tmp_replay_buffer) - 1:
                r = ig_reward * tmp_replay_buffer[i][-1]
                r_np = ig_reward_np * masks_np[i]
            else:
                # give reward only at that one game step, not all
                r = ig_reward * (tmp_replay_buffer[i][-1] - tmp_replay_buffer[i + 1][-1])
                r_np = ig_reward_np * (masks_np[i] - masks_np[i + 1])
            tmp_replay_buffer[i].append(r)
            command_rewards_np.append(r_np)
        command_rewards_np = np.array(command_rewards_np)
        if report: 
            print(command_rewards_np[:, 0])

        # push experience into replay buffer
        for b in range(len(ig_reward_np)):
            is_prior = np.sum(command_rewards_np, 0)[b] > 0.0
            mem = []
            for i in range(len(tmp_replay_buffer)):
                batch_description_list, batch_chosen_indices, batch_chosen_ctrlf_indices, batch_graph_node_vocabulary, batch_graph_relation_vocabulary, batch_graph_triplets, _, batch_rewards = tmp_replay_buffer[i]
                mem.append([copy.deepcopy(batch_description_list[b]), 
                            copy.deepcopy(quest_list[b]), 
                            batch_chosen_indices[b], 
                            batch_chosen_ctrlf_indices[b], 
                            copy.deepcopy(batch_graph_node_vocabulary[b]), 
                            copy.deepcopy(batch_graph_relation_vocabulary[b]), 
                            copy.deepcopy(batch_graph_triplets[b]), 
                            copy.deepcopy(belief_buffer[i][b].cpu()) if belief_buffer[i][b] is not None else None,
                            batch_rewards[b]])
                if masks_np[i][b] == 0.0:
                    break
            agent.replay_memory.push(is_prior, mem)

        qa_acc = np.mean(qa_reward_np)
        ig_acc = np.mean(ig_reward_np)
        step_masks_np = np.sum(np.array(masks_np), 0)  # batch
        for i in range(len(qa_reward_np)):
            # if the answer is totally wrong, we assume it used all steps
            if qa_reward_np[i] == 0.0:
                step_masks_np[i] = agent.max_nb_steps_per_episode
        used_steps = np.mean(step_masks_np)

        running_avg_qa_acc.push(qa_acc)
        running_avg_ig_acc.push(ig_acc)
        running_avg_steps.push(used_steps)
        print_rewards = np.sum(np.mean(command_rewards_np, -1))

        if report: 
            print("-- OBS: %s" % (obs_strings[0].encode('utf-8')))
            print("-- PRED: %s" % (chosen_answer_strings[0].encode('utf-8')))
        # finish game
        agent.finish_of_episode(episode_no, batch_no, batch_size)

        time_2 = datetime.datetime.now()
        eastern_time = datetime.datetime.now(timezone('US/Eastern')).strftime("%b %d %Y %H:%M:%S")
        if report: 
            print("Episode: {:3d} | {:s} | time spent: {:s} | interaction loss: {:2.3f} | interaction qvalue: {:2.3f} | qa loss: {:2.3f} | rewards: {:2.3f} | qa acc: {:2.3f}/{:2.3f} | sufficient info: {:2.3f}/{:2.3f} | used steps: {:2.3f}".format(episode_no, eastern_time, str(time_2 - time_1).rsplit(".")[0], running_avg_ig_loss.get_avg(),  running_avg_ig_q_value.get_avg(), running_avg_qa_loss.get_avg(), print_rewards, qa_acc, running_avg_qa_acc.get_avg(), ig_acc, running_avg_ig_acc.get_avg(), running_avg_steps.get_avg()))
        if __save__:
            agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_ep" + str(episode_no) + "_model.pt")
        if not report or episode_no < agent.learn_start_from_this_episode:
            episode_no += batch_size
            batch_no += 1
            continue

        eval_qa_acc, eval_ig_acc, eval_used_steps = 0.0, 0.0, 0.0
        # evaluate
        if agent.run_eval:
            eval_qa_acc, eval_ig_acc, eval_used_steps = evaluate.evaluate(env, agent, "valid")
            env.split_reset("train")
            # if run eval, then save model by eval accucacy
            if eval_qa_acc >= best_qa_acc_so_far:
                best_qa_acc_so_far = eval_qa_acc
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
            curr_performance = eval_qa_acc
        else:
            if running_avg_qa_acc.get_avg() >= best_qa_acc_so_far:
                best_qa_acc_so_far = running_avg_qa_acc.get_avg()
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
            curr_performance = running_avg_qa_acc.get_avg()

        if prev_performance <= curr_performance:
            i_am_patient = 0
        else:
            i_am_patient += 1
        prev_performance = curr_performance

        # if patient >= patience, resume from checkpoint
        if agent.patience > 0 and i_am_patient >= agent.patience:
            if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
                print('reload from a good checkpoint...')
                agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt", load_partial_graph=False)
                agent.update_target_net()
                i_am_patient = 0

        # plot using visdom
        if config["general"]["visdom"] and not agent.debug_mode:
            viz_avg_ig_acc.append(running_avg_ig_acc.get_avg())
            viz_avg_qa_acc.append(running_avg_qa_acc.get_avg())
            viz_avg_ig_q_value.append(running_avg_ig_q_value.get_avg())
            viz_eval_ig_acc.append(eval_ig_acc)
            viz_eval_qa_acc.append(eval_qa_acc)
            viz_eval_steps.append(eval_used_steps)
            viz_avg_steps.append(running_avg_steps.get_avg())
            viz_x = np.arange(len(viz_avg_ig_acc)).tolist()

            if plt_win is None:
                plt_win = viz.line(X=viz_x, Y=viz_avg_ig_acc,
                                   opts=dict(title=agent.experiment_tag + "_train"),
                                   name="sufficient info")
                viz.line(X=viz_x, Y=viz_avg_qa_acc,
                         opts=dict(title=agent.experiment_tag + "_train"),
                         win=plt_win, update='append', name="qa")
            else:
                viz.line(X=[len(viz_avg_ig_acc) - 1], Y=[viz_avg_ig_acc[-1]],
                         opts=dict(title=agent.experiment_tag + "_train"),
                         win=plt_win,
                         update='append', name="sufficient info")
                viz.line(X=[len(viz_avg_qa_acc) - 1], Y=[viz_avg_qa_acc[-1]],
                         opts=dict(title=agent.experiment_tag + "_train"),
                         win=plt_win,
                         update='append', name="qa")

            if plt_q_value_win is None:
                plt_q_value_win = viz.line(X=viz_x, Y=viz_avg_ig_q_value,
                                   opts=dict(title=agent.experiment_tag + "_train_q_value"),
                                   name="sufficient info")
            else:
                viz.line(X=[len(viz_avg_ig_q_value) - 1], Y=[viz_avg_ig_q_value[-1]],
                         opts=dict(title=agent.experiment_tag + "_train_q_value"),
                         win=plt_q_value_win,
                         update='append', name="sufficient info")

            if plt_steps_win is None:
                plt_steps_win = viz.line(X=viz_x, Y=viz_avg_steps,
                                   opts=dict(title=agent.experiment_tag + "_train_step"),
                                   name="used steps")
            else:
                viz.line(X=[len(viz_avg_steps) - 1], Y=[viz_avg_steps[-1]],
                         opts=dict(title=agent.experiment_tag + "_train_step"),
                         win=plt_steps_win,
                         update='append', name="used steps")
            if agent.run_eval:
                if eval_plt_win is None:
                    eval_plt_win = viz.line(X=viz_x, Y=viz_eval_ig_acc,
                                            opts=dict(title=agent.experiment_tag + "_eval"),
                                            name="sufficient info")
                    viz.line(X=viz_x, Y=viz_eval_qa_acc,
                            opts=dict(title=agent.experiment_tag + "_eval"),
                            win=eval_plt_win, update='append', name="qa")
                else:
                    viz.line(X=[len(viz_eval_ig_acc) - 1], Y=[viz_eval_ig_acc[-1]],
                            opts=dict(title=agent.experiment_tag + "_eval"),
                            win=eval_plt_win,
                            update='append', name="sufficient info")
                    viz.line(X=[len(viz_eval_qa_acc) - 1], Y=[viz_eval_qa_acc[-1]],
                            opts=dict(title=agent.experiment_tag + "_eval"),
                            win=eval_plt_win,
                            update='append', name="qa")

                if eval_plt_steps_win is None:
                    eval_plt_steps_win = viz.line(X=viz_x, Y=viz_eval_steps,
                                    opts=dict(title=agent.experiment_tag + "_eval_step"),
                                    name="used steps")
                else:
                    viz.line(X=[len(viz_avg_steps) - 1], Y=[viz_eval_steps[-1]],
                            opts=dict(title=agent.experiment_tag + "_eval_step"),
                            win=eval_plt_steps_win,
                            update='append', name="used steps")

        # write accucacies down into file
        _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
                         "sufficient info": str(running_avg_ig_acc.get_avg()),
                         "qa": str(running_avg_qa_acc.get_avg()),
                         "sufficient qvalue": str(running_avg_ig_q_value.get_avg()),
                         "eval sufficient info": str(eval_ig_acc),
                         "eval qa": str(eval_qa_acc),
                         "eval steps": str(eval_used_steps),
                         "used steps": str(running_avg_steps.get_avg())})
        with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
            outfile.write(_s + '\n')
            outfile.flush()
        episode_no += batch_size
        batch_no += 1

if __name__ == '__main__':
    train()
