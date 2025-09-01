import heapq
from LLM_Interface import transformers_interface
import numpy as np
from env.thor_env_modified import ThorEnv
import json
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

from functools import reduce
from operator import mul

from utils import compute_cos_similarity

from enum import Enum

LLAMA_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_PATH = "/home/atkesonlab2/models/"

# Enum for when to check constraints
# After each step generated in every beam, after the whole beam has been generated, or not at all
class ConstraintCheck(Enum):
    EACH_STEP = 1
    EACH_BEAM = 2
    NEVER = 3

class Alfred_Agent:
    """
    Initialize the agent with the following:
        language model path
        task name
        constraints path to load
        environment agent is acting in
    """

    def __init__(
        self,
        language_model: str = LLAMA_PATH,
        load_llm=True,
        task: str = "Make a BLT",
        constraints: str = "",
        env=None,
        selection_method="sim",
        # task_graph_path = ""
    ):
        if load_llm:
            self.llm = transformers_interface(model_name=LLAMA_PATH, cuda_devices=[0])

        cross_task_narrations_path = "Narrations/cross_task_narrations.json"
        with open(cross_task_narrations_path, "r") as file:
            self.cross_task_narrations = json.load(file)

        self.text_embedd_model = SentenceTransformer(
            "multi-qa-mpnet-base-cos-v1", device="cuda"
        )
        self.prob_scaling_factor = 3
        self.selection_method = selection_method

        self.task = task
        self.constraints = constraints
        self.env = env
        self.step_similarity_threshold = 0.5

        self.task_graph_path = None


    def set_env(self, env: ThorEnv):
        self.env = env

    def set_task(self, task: str):
        self.task = task
    
    def set_task_graph_path(self, task_graph_path):
        self.task_graph_path = task_graph_path
        with open(self.task_graph_path, "r") as file:
            self.task_graph = json.load(file)
        self.task_graph_keys = list(self.task_graph.keys())
        self.task_graph_embedds = self.text_embedd_model.encode(self.task_graph_keys)

    def retrieve_constraints(self, actions: List[str] = []):
        return [
            "pick up knife before slicing anything",
            "slice item before adding it to dish",
            "cannot slice item without picking up knife",
            "do not repeat previous actions",
            "goto an object before any interaction with the object"
            # "goto an object before picking it up"
        ]

    def generate_next_action_prompt(
        self, beam_actions: List[str] = None
    ):
        prompt = "USER: I am attempting to complete the task: " + self.task + ". The steps completed so far are: \n"
        for i, step in enumerate(self.env.action_history):
            prompt += "Step {}: ".format(i + 1) + step + "\n"

        if self.env.last_failed_action != None:
            prompt += "The last action, " + self.env.last_failed_action + " failed.\n"

        if (beam_actions is not None):
            prompt += "Below is a list of proposed actions"
            for i, step in enumerate(beam_actions):
                prompt += "Step {}: ".format(i + len(self.env.action_history) + 1) + step["action_str"] + "\n"
        prompt += "Is [action] an appropriate next step to complete the task? ASSITANT: "

        return prompt


    def validate_each_action_prompt(self, beam_actions, new_action):
        final_prompt = (
            "A person has been assigned the following task to complete: " + self.task + "\n"
        )

        if (len(self.env.action_history) > 0):
            final_prompt += "Here is a breakdown of actions the person has taken so far:\n"
            for act in self.env.action_history:
                final_prompt += act[0] + "\n"

            for act in beam_actions:
                final_prompt += act["action_str"] + "\n"

        final_prompt += "Here is the next proposed action:\n"
        final_prompt += new_action["action_str"] + "\n"

        final_prompt += "Based on the action history, does the new proposed action satisfy the constraint \"[action]\"? Answer yes or no."

        return final_prompt
    
    def validate_each_beam_prompt(self, new_actions):
        final_prompt = (
            "A person has been assigned the following task to complete: " + self.task + "\n"
        )

        if (len(self.env.action_history) > 0):
            final_prompt += "Here is a breakdown of actions the person has taken so far:\n"
            for act in self.env.action_history:
                final_prompt += act[0] + "\n"

        final_prompt += "Here is a sequence of proposed next actions to take:\n"
        for act in new_actions:
            final_prompt += act["action_str"] + "\n"

        final_prompt += "Based on the action history, does the new proposed sequence of actions satisfy the constraint \"[action]\"? Answer yes or no."

        return final_prompt


    def action_probs_yn(self, prompt, actions):
        probabilities = [
            self.llm.yes_no_question_chat(prompt.replace("[action]", action["action_str"]))[0]
            for action in actions
        ]
        return probabilities
    
    def constraint_probs_yn(self, prompt):
        probabilities = [
            self.llm.yes_no_question_chat(prompt.replace("[action]", constraint))[0]
            for constraint in self.retrieve_constraints()
        ]
        return probabilities

    """
    Given a prompt and a set of predicates representing the environment, outputs the next most probable actions.
    Input:
            prompt: str - task to complete
            predicates: List[str] - environment state
            possible_actions: List[Dict] - potential actions to take
            action_history: List[str] - actions already taken
            top_k: int - number of top actions to return
    Output:
            List[Tuple[str, float]] - top-k actions with their probabilities
    """

    def determine_action_probabilities(
        self,
        beam_actions: List[str],
        top_k: int,
        use_constraints=ConstraintCheck.NEVER
    ) -> List[Dict]:
        action_prompt = self.generate_next_action_prompt(beam_actions)

        possible_actions = self.env.generate_possible_actions()

        action_probabilities = self.action_probs_yn(action_prompt, possible_actions)
        action_probabilities = np.array(action_probabilities)
        action_probabilities = action_probabilities / np.sum(action_probabilities)


        if use_constraints == ConstraintCheck.EACH_STEP:
            constraint_probabilities = []
            for action in possible_actions:
                action_constraint_probs = self.constraint_probs_yn(
                    self.validate_each_action_prompt(beam_actions=beam_actions, new_action=action)
                )
                constraint_prob = reduce(mul, action_constraint_probs)
                constraint_probabilities.append(constraint_prob)
        else:
            constraint_probabilities = [1.0 for x in action_probabilities]

        final_probabilities = [x * y for (x, y) in zip (action_probabilities, constraint_probabilities)]

        action_prob_pairs = []
        for action, prob in zip(possible_actions, final_probabilities):
            action_copy = action.copy()
            action_copy["probability"] = prob
            action_prob_pairs.append(action_copy)

        return heapq.nlargest(top_k, action_prob_pairs, key=lambda x: x["probability"])

    """
    Performs beam search for a sequence of actions.
    Input:
            predicates: List[str] - environment state
            possible_actions: List[Dict] - potential actions to take
            planning_horizon: int - number of steps to plan ahead
    Output:
            List[str] - action sequence with the highest probability
    """

    '''
    format of generate_possible_actions
    {
    "obj": ...,
    "action": ...,
    "action_str": ...,
    "probability": ...
    }
    '''

    def beam_search(
        self,
        planning_horizon: int,
        top_k: int,
        use_constraints: ConstraintCheck = ConstraintCheck.NEVER
    ) -> List[str]:
        beams = [(1.0, [])]  # does all already executed actions with prob 1

        for _ in range(planning_horizon):  # length of beam
            new_beams = []  # new beams

            for prob, beam_actions in beams:
                top_actions = self.determine_action_probabilities(
                    beam_actions,
                    top_k,
                    use_constraints,
                )
                for action in top_actions:
                    new_beams.append(
                        (prob * action["probability"], beam_actions + [action])
                    )
                              
            beams = heapq.nlargest(top_k, new_beams, key=lambda x: x[0])
            total_prob = sum(prob for prob, _ in beams)
            beams = [(prob / total_prob, actions) for prob, actions in beams]
        
        #after all beams are generated, check constraints for each beam
        if use_constraints == ConstraintCheck.EACH_BEAM:
            beams_with_constraints = []
            for prob, beam_actions in beams:
                constraint_probabilities = self.llm.action_probs_yn(
                    self.validate_each_beam_prompt(beam_actions),
                )
                constraint_prob = reduce(mul, constraint_probabilities)
                beams_with_constraints.append((constraint_prob * prob, beam_actions))
            beams = beams_with_constraints
        
        total_prob = sum(prob for prob, _ in beams)
        beams = [(prob / total_prob, actions) for prob, actions in beams]

        # Return the action sequence with the highest probability
        print(
            "ACTION PROB: ",
            max(beams, key=lambda x: x[0])[0],
            " ACTIONS: ",
            [action["action_str"] for action in max(beams, key=lambda x: x[0])[1]],
        )
        return max(beams, key=lambda x: x[0])[1]

    def select_action(self):        
        top_actions = self.beam_search(
            planning_horizon=3,
            top_k=3,
            use_constraints=ConstraintCheck.EACH_STEP
        )

        return top_actions[0]




if __name__ == "__main__":
    pass
