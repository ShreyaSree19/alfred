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
        task: str = "",
        constraints: str = "",
        env=None,
        selection_method="sim",
        # task_graph_path = ""
    ):
        if load_llm:
            self.llm = transformers_interface(model_name=LLAMA_PATH, cuda_devices=[0])


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
    
    #hard coded constraints
    def retrieve_constraints(self, actions: List[str] = []):
        return [
            # "pick up knife before slicing anything",
            # "slice item before adding it to dish",
            # "cannot slice item without picking up knife",
            # "do not repeat previous actions",
            # "goto an object before any interaction with the object"
            # "goto an object before picking it up"
        ]

    def generate_next_action_prompt(
        self, beam_actions: List[str] = None, mcq = False
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
        if mcq:
            prompt += "What's the best action to take? ASSITANT: "
        else:
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


    def action_probs_yn(self, prompt, actions):
        probabilities = [
            self.llm.yes_no_question_chat(prompt.replace("[action]", action["action_str"]))[0]
            for action in actions
        ]
        return probabilities
    
    def action_probs_mcq(self, prompt, actions):
        # getting only the yes probability
        action_strs = [action["action_str"] for action in actions]
        probabilities = self.llm.action_probs_mcq(prompt , action_strs)
        return probabilities
    
    def constraint_probs_yn(self, prompt):
        probabilities = [
            self.llm.yes_no_question_chat(prompt.replace("[action]", constraint))[0]
            for constraint in self.retrieve_constraints()
        ]
        return probabilities

 
    def determine_action_probabilities(
        self,
        beam_actions: List[str],
        top_k: int,
        use_constraints=ConstraintCheck.NEVER,
        top_n_mcq: int = 5,
        use_chat: bool = True
    ) -> List[Dict]:
        """
        Determine action probabilities using optional two-stage LLM scoring:
        Stage 1: Yes/No probabilities for all actions
        Stage 2: MCQ probabilities for top-N actions

        Returns top-k actions with combined probability (Yes/No * constraints * MCQ)
        """

        action_prompt = self.generate_next_action_prompt(beam_actions)
        possible_actions = self.env.generate_possible_actions()
        # print(possible_actions)

        # Stage 1: Yes/No scoring for all actions
        yes_probs = self.action_probs_yn(action_prompt, possible_actions)
        yes_probs = np.array(yes_probs)
        yes_probs = yes_probs / np.sum(yes_probs)

        # Pick top-N candidates for MCQ
        top_indices = np.argsort(yes_probs)[::-1][:top_n_mcq]
        top_actions_mcq = [possible_actions[i] for i in top_indices]

        # Stage 2: MCQ scoring on top-N actions
        mcq_prompt = self.generate_next_action_prompt(beam_actions, True)
        mcq_probs = self.action_probs_mcq(action_prompt, top_actions_mcq)
        mcq_probs = np.array(mcq_probs)
        mcq_probs = mcq_probs / np.sum(mcq_probs)

        # Combine Yes/No and MCQ probabilities
        combined_probs = mcq_probs

        # Apply constraints if needed
        if use_constraints == ConstraintCheck.EACH_STEP:
            for i, action in enumerate(top_actions_mcq):
                constraint_probs = self.constraint_probs_yn(
                    self.validate_each_action_prompt(beam_actions=beam_actions, new_action=action)
                )
                constraint_prob = reduce(mul, constraint_probs)
                combined_probs[i] *= constraint_prob

        # Return top-k actions
        action_prob_pairs = []
        for action, prob in zip(top_actions_mcq, combined_probs):
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
            If use_beam_search = False, performs greedy one-step action selection.
            If use_beam_search = True, performs multi-step beam search planning.
    Output:
            List[str] - action sequence with the highest probability    """

    # one step picks the best one in the immediate children (horizontal layer, planning_horizon is 1
    # beam search picks the one that has the best result planning_horizon vertical levels down

    def beam_search(
    self,
    planning_horizon: int,
    top_k: int,
    use_constraints: ConstraintCheck,
    use_beam_search: bool):
        #One-step greedy planning
        if not use_beam_search:
            top_actions = self.determine_action_probabilities(
                beam_actions=[],
                top_k=top_k,
                use_constraints=use_constraints,
            )
            print("TOP GREEDY ACTIONS:", [a["action_str"] for a in top_actions])
            best_action = max(top_actions, key=lambda a: a["probability"])
            print("GREEDY ACTION:", best_action["action_str"])
            return [best_action]

        # Multi-step beam search planning
        beams = [(1.0, [])]

        for _ in range(planning_horizon):
            new_beams = []
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

        best_prob, best_actions = max(beams, key=lambda x: x[0])
        print("BEAM ACTIONS:", [a["action_str"] for a in best_actions])
        return best_actions


    def select_action(self):
        top_actions = self.beam_search(
            planning_horizon=3,
            top_k=3,
            use_constraints=ConstraintCheck.NEVER,
            use_beam_search=False  # toggle here
        )
        return top_actions[0]



if __name__ == "__main__":
    pass
