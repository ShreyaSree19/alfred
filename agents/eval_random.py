import os
import json
import random
import traceback
from env.thor_env_modified import ThorEnv

class DummyArgs:
    def __init__(self):
        self.reward_config = "../models/config/rewards.json"
        self.record_continuous_video = False
        self.record_failed_trials = False
        self.max_episode_length = 10
        self.controller_type = "STOCHASTIC"
        self.visibility_distance = 1.5
        self.record_timeline = False
        self.random_init = False
        self.recording = False
        self.randomize_start = False
        self.goal_desc_human = ""
        self.goal_desc_task = ""
        self.eval = True
        self.save_frames = False


def random_select_action(env):
    possible_actions = env.generate_possible_actions()
    valid_actions = []
    for a in possible_actions:
        if not a or "action" not in a:
            continue
        if a["action"] == "GoToObject" and (not a.get("obj") or "objectId" not in a["obj"]):
            continue
        valid_actions.append(a)
    if not valid_actions:
        return {"action_str": "done"}
    return random.choice(valid_actions)


def single_eval_random(env, max_steps=10):
    print("INIT SUCCESS", env.get_goal_satisfied())
    for step in range(max_steps):
        action_dict = random_select_action(env)
        if action_dict["action_str"] == "done":
            break

        action_type = action_dict["action"]
        thor_args = {
            "action": action_type,
            "action_str": action_dict.get("action_str", "")
        }

        # Pass object info if it exists
        if "obj" in action_dict and action_dict["obj"]:
            thor_args["object_id"] = action_dict["obj"].get("objectId", "")
            thor_args["obj"] = action_dict["obj"]

        # For PutObject, include receptacle
        if action_type == "PutObject" and "receptacle" in action_dict:
            thor_args["receptacleObjectId"] = action_dict["receptacle"]

        try:
            env.to_thor_api_exec(**thor_args)

        # Unity crash recovery
        except TimeoutError:
            print(f"[WARN] Timeout in action {action_type}. Restarting environment to recover...")
            env.stop_controller()  # terminate dead Unity
            env = ThorEnv()  # restart new THOR instance
            print("[INFO] Environment restarted successfully.")
            return False
            
        except Exception as e:
            print(f"[WARN] Skipping action {action_type} due to error: {e}")
            continue
               
        # Check if task succeeded
        if getattr(env, "task", None) is not None:
            success = env.get_goal_satisfied()        
    return success



def eval_random_baseline(tasks_dir, max_steps=10):
    env = ThorEnv()
    total = 0
    successes = 0
    dummy_args = DummyArgs()
    #around 250 total
    for root, _, files in os.walk(tasks_dir):
        for file in files:
            print("Here", total)

            if not file.endswith(".json"):
                continue

            task_path = os.path.join(root, file)
            with open(task_path, "r") as f:
                task_data = json.load(f)
            print(task_data["turk_annotations"]["anns"][0]["task_desc"])

            scene_num = task_data["scene"]["scene_num"]
            scene_name = f"FloorPlan{scene_num}"
            
            env.reset(scene_name)
            env.set_task(task_data, dummy_args)

            success = single_eval_random(env, max_steps=max_steps)
            successes += int(success)

            print(f"[{total}] {file} → {'SUCCESS' if success else 'FAIL'}")
            total += 1

        if total > 200:
            break

    env.stop()
    success_rate = successes / total if total > 0 else 0
    print(f"\nRandom baseline success rate: {success_rate * 100:.2f}%")
    return success_rate

if __name__ == "__main__":
    TASKS_DIR = "../data/json_2.1.0/valid_seen/"
    eval_random_baseline(TASKS_DIR, max_steps=10)