import json

from gamified_squad import GamifiedSquad
from agent import CustomAgent
import generic
import evaluate


def train():

    config = generic.load_config()
    env = GamifiedSquad(config)
    env.split_reset("train")
    agent = CustomAgent(config, env.has_token_set)

    output_dir = "."
    data_dir = "."
    json_file_name = agent.experiment_tag.replace(" ", "_")
    # load model from checkpoint


    agent.load_pretrained_graph_generation_model(data_dir + "/" + agent.load_graph_generation_model_from_tag + ".pt")
    agent.load_pretrained_model(agent.load_from_tag + ".pt", load_partial_graph=False)
    agent.update_target_net()

    eval_qa_acc, eval_ig_acc, eval_used_steps = evaluate.evaluate(env, agent, "test")


    # write accucacies down into file
    _s = json.dumps({"time spent": "0",
                     "sufficient info": "0",
                     "qa": "0",
                     "sufficient qvalue": "0",
                     "eval sufficient info": str(eval_ig_acc),
                     "eval qa": str(eval_qa_acc),
                     "eval steps": str(eval_used_steps),
                     "used steps": "0"})
    with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
        outfile.write(_s + '\n')
        outfile.flush()

if __name__ == '__main__':
    train()
