import copy
import json
import copy

from gamified_squad import GamifiedSquad
from ensemble_agent import EnsembleAgent
import generic
import evaluate


def train():

    config = generic.load_config()
    env = GamifiedSquad(config)
    env.split_reset("train")

    concat_tags = config["checkpoint"]["load_from_tag"]
    tag_list = concat_tags.split("|||")

    agents = []
    for i, g in enumerate(["cooccur", "relative_position", "srl", "gata"]):
        __config = copy.copy(config)

        __config["general"]["enable_graph_input"] = g
        if g in ["gata"]:
            __config["general"]["node_capacity"] = 64
            __config["general"]["relation_capacity"] = 16
            __config["checkpoint"]["load_graph_generation_model_from_tag"] = 'gamify_gata_pretrained_models/pretrain_3928a10_model'
        elif g == "srl":
            __config["general"]["node_capacity"] = 64
            __config["general"]["relation_capacity"] = 32
        else:
            __config["general"]["node_capacity"] = 64
            __config["general"]["relation_capacity"] = 11

        __config["checkpoint"]["load_from_tag"] = tag_list[i]
        __agent = EnsembleAgent(__config, env.has_token_set)
        agents.append(__agent)


    output_dir = "."
    data_dir = "."
    json_file_name = agents[0].experiment_tag.replace(" ", "_")
    # load model from checkpoint


    for i in range(len(agents)):
        agents[i].load_pretrained_graph_generation_model(data_dir + "/" + agents[i].load_graph_generation_model_from_tag + ".pt")
        agents[i].load_pretrained_model(agents[i].load_from_tag + ".pt", load_partial_graph=False)
        agents[i].update_target_net()

    eval_qa_acc, eval_ig_acc, eval_used_steps = evaluate.ensemble_evaluate(env, agents, "test")


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
