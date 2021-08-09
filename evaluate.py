import numpy as np
import generic
import torch


def evaluate(env, agent, valid_test="valid", verbose=False):
    with torch.no_grad():
        env.seed(42)
        env.split_reset(valid_test)
        agent.eval()
        print_qa_acc, print_ig_acc, print_steps = [], [], []
        to_print, counter = [], 0

        while(True):
            obs, infos = env.reset(random=False)
            batch_size = len(obs)
            agent.init(obs, infos)
            quest_list = agent.get_game_quest_info(infos)
            agent.kg.push_batch_question(quest_list, [item["q_srl"] for item in infos])
            previous_belief = None
            if agent.use_gt_graph:
                for i in range(batch_size):
                    full_obs = infos[i]["full_obs"]
                    full_srl = infos[i]["full_srl"]
                    for j in range(len(full_obs)):
                        agent.kg.push_one(i, full_obs[j], full_srl[j])

            input_quest, input_quest_mask, quest_id_list = agent.get_agent_inputs(quest_list)
            previous_dynamics = None

            tmp_replay_buffer = []
            prev_commands = ["restart" for _ in range(batch_size)]
            for _ in range(agent.max_nb_steps_per_episode):
                commands, replay_info, current_dynamics, current_belief = agent.act_greedy(obs, infos, input_quest, input_quest_mask, quest_id_list, prev_commands, previous_dynamics, previous_belief)
                tmp_replay_buffer.append(replay_info)
                obs, infos = env.step(commands)
                prev_commands = commands
                previous_dynamics = current_dynamics
                previous_belief = current_belief

                still_running = generic.to_np(replay_info[-1])
                if np.sum(still_running) == 0:
                    break

            # The agent has exhausted all steps, now answer question.
            chosen_head_tails = agent.answer_question_act(agent.naozi.get(), quest_list, current_belief)  # batch
            chosen_head_tails_np = generic.to_np(chosen_head_tails)
            chosen_answer_strings = generic.get_answer_strings(agent.naozi.get(), chosen_head_tails_np, agent.bert_tokenizer, agent.special_token_ids)
            answer_strings = [item["a_string"] for item in infos]
            answer_token_ids = [item["a"] for item in infos]
            masks_np = [generic.to_np(item[-1]) for item in tmp_replay_buffer]

            qa_reward_np = generic.get_qa_reward(chosen_answer_strings, answer_strings)
            ig_reward_np = generic.get_sufficient_info_reward(agent.naozi.get(), answer_token_ids)

            if verbose:
                for i in range(batch_size):
                    to_print.append(str(counter) + " -------------------------------------------- ig: " + str(ig_reward_np[i]) + ", qa: " + str(qa_reward_np[i]))
                    to_print.append("Q: %s " % (agent.bert_tokenizer.decode(infos[0]["q"])))
                    to_print.append("OBS: %s " % (agent.bert_tokenizer.decode(agent.naozi.get(i))))
                    to_print.append("PRED: %s " % (chosen_answer_strings[i]))
                    to_print.append("GT: %s " % (" | ".join(answer_strings[i])))
                    to_print.append("")
                    counter += 1

            step_masks_np = np.sum(np.array(masks_np), 0)
            for i in range(len(qa_reward_np)):
                # if the answer is totally wrong, we assume it used all steps
                if qa_reward_np[i] == 0.0:
                    step_masks_np[i] = agent.max_nb_steps_per_episode
            print_qa_acc += qa_reward_np.tolist()
            print_ig_acc += ig_reward_np.tolist()
            print_steps += step_masks_np.tolist()
            if env.batch_pointer == 0:
                break

        if verbose:
            with open(agent.experiment_tag + "_output.txt", "w") as f:
                f.write("\n".join(to_print))
        print("===== Eval =====: qa acc: {:2.3f} | correct state: {:2.3f} | used steps: {:2.3f}".format(np.mean(np.array(print_qa_acc)), np.mean(np.array(print_ig_acc)), np.mean(np.array(print_steps))))
        return np.mean(np.array(print_qa_acc)), np.mean(np.array(print_ig_acc)), np.mean(np.array(print_steps))


def evaluate_observation_infomax(env, agent, valid_test="valid"):
    env.split_reset(valid_test)
    agent.eval()
    total_valid_loss = []
    total_valid_accuracy = []

    while(True):

        positive_samples, negative_samples = env.get_batch()
        with torch.no_grad():
            valid_loss, acc = agent.get_observation_infomax_loss(positive_samples, negative_samples, evaluate=True)

        total_valid_loss = total_valid_loss + valid_loss
        total_valid_accuracy = total_valid_accuracy + acc
        if env.batch_pointer == 0:
            break
    report_loss = np.mean(np.array(total_valid_loss))
    report_accuracy = np.mean(np.array(total_valid_accuracy))
    print("evaluating " + valid_test + " set, loss: " + str(report_loss) + ", acc: " + str(report_accuracy))
    return report_loss, report_accuracy


def ensemble_evaluate(env, agents, valid_test="valid"):
    with torch.no_grad():
        env.seed(42)
        env.split_reset(valid_test)
        num_agents = len(agents)
        for i in range(num_agents):
            agents[i].eval()
        print_qa_acc, print_ig_acc, print_steps = [], [], []

        while(True):
            obs, infos = env.reset(random=False)
            batch_size = len(obs)
            quest_list = agents[0].get_game_quest_info(infos)
            input_quest, input_quest_mask, quest_id_list = agents[0].get_agent_inputs(quest_list)

            tmp_replay_buffer = []
            previous_dynamics = []
            previous_belief = []
            for i in range(num_agents):
                agents[i].init(obs, infos)
                agents[i].kg.push_batch_question(quest_list, [item["q_srl"] for item in infos])
                if agents[i].use_gt_graph:
                    for i in range(batch_size):
                        full_obs = infos[i]["full_obs"]
                        full_srl = infos[i]["full_srl"]
                        for j in range(len(full_obs)):
                            agents[i].kg.push_one(i, full_obs[j], full_srl[j])
                previous_belief.append(None)
                previous_dynamics.append(None)

            prev_commands = ["restart" for _ in range(batch_size)]
            for _ in range(agents[0].max_nb_steps_per_episode):

                overall_a_rank, overall_c_rank = None, None
                overall_ctrlf_word_mask = None
                for i in range(num_agents):
                    action_rank, ctrlf_rank, ctrlf_word_mask, current_dynamics, current_belief, replay_info = agents[i].get_ranks_greedy(obs, infos, input_quest, input_quest_mask, quest_id_list, prev_commands, previous_dynamics[i], previous_belief[i])
                    if i == 0:
                        tmp_replay_buffer.append(replay_info)
                    if overall_a_rank is None:
                        overall_a_rank = action_rank
                        overall_c_rank = ctrlf_rank
                        overall_ctrlf_word_mask = ctrlf_word_mask
                    else:
                        overall_a_rank = overall_a_rank + action_rank
                        overall_c_rank = overall_c_rank + ctrlf_rank
                    previous_dynamics[i] = current_dynamics
                    previous_belief[i] = current_belief

                action_indices = agents[0].choose_maxQ_command(overall_a_rank)
                ctrlf_indices = agents[0].choose_maxQ_command(overall_c_rank, overall_ctrlf_word_mask)
                commands = agents[0].generate_commands(action_indices, ctrlf_indices)    
                
                for i in range(num_agents):
                    for j in range(batch_size):
                        if commands[j] == "stop":
                            agents[i].not_finished_yet[j] = 0.0
                    agents[i].prev_actions.append(commands)

                obs, infos = env.step(commands)
                prev_commands = commands
                still_running = generic.to_np(replay_info[-1])
                if np.sum(still_running) == 0:
                    break

            # The agent has exhausted all steps, now answer question.
            overall_qa_rank, overall_qa_mask = None, None
            for i in range(num_agents):
                qa_rank, qa_mask = agents[i].get_qa_ranks_greedy(agents[0].naozi.get(), quest_list, previous_belief[i])  # batch
                if overall_qa_rank is None:
                    overall_qa_rank = qa_rank
                    overall_qa_mask = qa_mask
                else:
                    overall_qa_rank = overall_qa_rank + qa_rank

            chosen_head_tails = agents[0].point_maxq_position(overall_qa_rank, overall_qa_mask)  # batch x 2
            chosen_head_tails_np = generic.to_np(chosen_head_tails)
            chosen_answer_strings = generic.get_answer_strings(agents[0].naozi.get(), chosen_head_tails_np, agents[0].bert_tokenizer, agents[0].special_token_ids)

            answer_strings = [item["a_string"] for item in infos]
            answer_token_ids = [item["a"] for item in infos]
            masks_np = [generic.to_np(item[-1]) for item in tmp_replay_buffer]

            qa_reward_np = generic.get_qa_reward(chosen_answer_strings, answer_strings)
            ig_reward_np = generic.get_sufficient_info_reward(agents[0].naozi.get(), answer_token_ids)

            step_masks_np = np.sum(np.array(masks_np), 0)
            for i in range(len(qa_reward_np)):
                # if the answer is totally wrong, we assume it used all steps
                if qa_reward_np[i] == 0.0:
                    step_masks_np[i] = agents[0].max_nb_steps_per_episode
            print_qa_acc += qa_reward_np.tolist()
            print_ig_acc += ig_reward_np.tolist()
            print_steps += step_masks_np.tolist()
            if env.batch_pointer == 0:
                break

        print("===== Eval =====: qa acc: {:2.3f} | correct state: {:2.3f} | used steps: {:2.3f}".format(np.mean(np.array(print_qa_acc)), np.mean(np.array(print_ig_acc)), np.mean(np.array(print_steps))))
        return np.mean(np.array(print_qa_acc)), np.mean(np.array(print_ig_acc)), np.mean(np.array(print_steps))
