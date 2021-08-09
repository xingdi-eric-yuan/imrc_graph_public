import torch
import datetime
import os
import json
import numpy as np

from observation_infomax_dataset import ObservationInfomaxData
from agent import CustomAgent
import generic
import evaluate


def train():

    time_1 = datetime.datetime.now()
    config = generic.load_config()
    env = ObservationInfomaxData(config)
    env.split_reset("train")
    agent = CustomAgent(config, set())
    ave_train_loss = generic.HistoryScoreCache(capacity=500)

    # visdom
    if config["general"]["visdom"]:
        import visdom
        viz = visdom.Visdom()
        loss_win = None
        acc_win = None
        viz_loss, viz_eval_loss, viz_eval_acc = [], [], []

    episode_no = 0
    batch_no = 0

    output_dir = "."
    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_training_loss_so_far, best_eval_loss_so_far = 10000.0, 10000.0
    # load model from checkpoint
    if agent.load_pretrained:
        if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
            agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt", load_partial_graph=False)

    try:
        while(True):
            if episode_no > agent.max_episode:
                break
            agent.train()
            positive_samples, negative_samples = env.get_batch()
            training_losses, _ = agent.get_observation_infomax_loss(positive_samples, negative_samples)

            batch_size = len(positive_samples)
            report = agent.report_frequency > 0 and (episode_no % agent.report_frequency <= max(episode_no - batch_size, 0) % agent.report_frequency)
            for _loss in training_losses:
                ave_train_loss.push(_loss)

            episode_no += batch_size
            batch_no += 1
            agent.finish_of_episode(episode_no, batch_no, batch_size)
            
            if not report:
                continue

            eval_loss, eval_acc = 100000.0, 0
            if agent.run_eval:
                eval_loss, eval_acc = evaluate.evaluate_observation_infomax(env, agent, "valid")
                env.split_reset("train")
                # if run eval, then save model by eval accuracy
                if eval_loss < best_eval_loss_so_far:
                    best_eval_loss_so_far = eval_loss
                    agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
            else:
                loss = ave_train_loss.get_avg()
                if loss < best_training_loss_so_far:
                    best_training_loss_so_far = loss
                    agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")

            time_2 = datetime.datetime.now()
            print("Episode: {:3d} | time spent: {:s} | loss: {:2.3f} | valid loss: {:2.3f} | valid acc: {:2.3f}".format(episode_no, str(time_2 - time_1).rsplit(".")[0], ave_train_loss.get_avg(), eval_loss, eval_acc))

            # plot using visdom
            if config["general"]["visdom"] and not agent.debug_mode:
                viz_loss.append(ave_train_loss.get_avg())
                viz_eval_loss.append(eval_loss)
                viz_eval_acc.append(eval_acc)
                viz_x = np.arange(len(viz_loss)).tolist()

                if loss_win is None:
                    loss_win = viz.line(X=viz_x, Y=viz_loss,
                                    opts=dict(title=agent.experiment_tag + "_loss"),
                                    name="training loss")

                    viz.line(X=viz_x, Y=viz_eval_loss,
                            opts=dict(title=agent.experiment_tag + "_eval_loss"),
                            win=loss_win,
                            update='append', name="eval loss")
                else:
                    viz.line(X=[len(viz_loss) - 1], Y=[viz_loss[-1]],
                            opts=dict(title=agent.experiment_tag + "_loss"),
                            win=loss_win,
                            update='append', name="training loss")

                    viz.line(X=[len(viz_eval_loss) - 1], Y=[viz_eval_loss[-1]],
                            opts=dict(title=agent.experiment_tag + "_eval_loss"),
                            win=loss_win,
                            update='append', name="eval loss")

                if acc_win is None:
                    acc_win = viz.line(X=viz_x, Y=viz_eval_acc,
                                    opts=dict(title=agent.experiment_tag + "_eval_acc"),
                                    name="eval accuracy")
                else:
                    viz.line(X=[len(viz_loss) - 1], Y=[viz_eval_acc[-1]],
                            opts=dict(title=agent.experiment_tag + "_eval_acc"),
                            win=acc_win,
                            update='append', name="eval accuracy")

            # write accuracies down into file
            _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
                            "loss": str(ave_train_loss.get_avg()),
                            "eval loss": str(eval_loss),
                            "eval accuracy": str(eval_acc)})
            with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
                outfile.write(_s + '\n')
                outfile.flush()
    
    # At any point you can hit Ctrl + C to break out of training early.
    except KeyboardInterrupt:
        print('--------------------------------------------')
        print('Exiting from training early...')
    if agent.run_eval:
        if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
            print('Evaluating on test set and saving log...')
            agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt", load_partial_graph=False)
        eval_loss, eval_acc = evaluate.evaluate_observation_infomax(env, agent, "test")


if __name__ == '__main__':
    train()
