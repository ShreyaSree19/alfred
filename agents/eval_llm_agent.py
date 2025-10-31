import os
import json
import traceback
from env.thor_env_modified import ThorEnv
from saycan_agent import Alfred_Agent


class DummyArgs:
    def __init__(self):
        self.reward_config = "../models/config/rewards.json"
        self.record_continuous_video = False
        self.record_failed_trials = False
        self.max_episode_length = 10
        self.controller_type = "DETERMINISTIC"
        self.visibility_distance = 1.5
        self.record_timeline = False
        self.random_init = False
        self.recording = False
        self.randomize_start = False
        self.goal_desc_human = ""
        self.goal_desc_task = ""
        self.eval = True
        self.save_frames = False


def single_eval_agent(env, agent, max_steps=10):
    # Run one evaluation episode with Alfred_Agent
    print("Initial goal satisfied:", env.get_goal_satisfied())

    for step in range(max_steps):
        print("Step: ", step)
        action_dict = agent.select_action()  
        if action_dict["action_str"] == "done":
            break

        action_type = action_dict["action"]
        thor_args = {
            "action": action_type,
            "action_str": action_dict.get("action_str", "")
        }

        if "obj" in action_dict and action_dict["obj"]:
            thor_args["object_id"] = action_dict["obj"].get("objectId", "")
            thor_args["obj"] = action_dict["obj"]

        if action_type == "PutObject" and "receptacle" in action_dict:
            thor_args["receptacleObjectId"] = action_dict["receptacle"]

        try:
            env.to_thor_api_exec(**thor_args)

        except TimeoutError:
            print(f"[WARN] Timeout in action {action_type}. Restarting environment to recover...")
            env.stop_controller()
            env = ThorEnv()
            print("[INFO] Environment restarted successfully.")
            return False

        except Exception as e:
            print(f"[WARN] Skipping action {action_type} due to error: {e}")
            continue

        # Check if task succeeded
        if getattr(env, "task", None) is not None:
            try:
                success = env.get_goal_satisfied()
            except ValueError as e:
                print(f"[Warning] Skipping goal check: {e}")
                return False

    return success


def eval_agent_baseline(tasks_dir, max_steps=10):
    env = ThorEnv()
    total, successes = 0, 0
    dummy_args = DummyArgs()
    agent = Alfred_Agent(env=env)
    print("Here")

    for root, _, files in os.walk(tasks_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            total += 1
            task_path = os.path.join(root, file)
            with open(task_path, "r") as f:
                task_data = json.load(f)

            scene_num = task_data["scene"]["scene_num"]
            scene_name = f"FloorPlan{scene_num}"

            task_desc = task_data["turk_annotations"]["anns"][0]["task_desc"]
            scene_num = task_data['scene']['scene_num']
            object_poses = task_data['scene']['object_poses']
            dirty_and_empty = task_data['scene']['dirty_and_empty']
            object_toggles = task_data['scene']['object_toggles']
            init_action = task_data['scene']['init_action']
            init_action.pop('rotateOnTeleport')
            init_action['standing'] = True
            print(task_desc)
            env.set_task(task_data, dummy_args)
            env.reset(scene_name)

            agent.set_task(task_desc)
            agent.set_env(env)


            # env.restore_scene(object_poses, object_toggles, dirty_and_empty)

            # # initialize to start position
            # env.step(dict(task_data['scene']['init_action']))

            success = single_eval_agent(env, agent, max_steps=max_steps)
            successes += int(success)
            print(f"[{total}] {file} → {'SUCCESS' if success else 'FAIL'}")

        if total > 200:
            break

    env.stop()
    success_rate = successes / total if total > 0 else 0
    print(f"\nAgent success rate: {success_rate * 100:.2f}%")
    return success_rate


if __name__ == "__main__":
    TASKS_DIR = "../data/json_2.1.0/valid_seen/"
    eval_agent_baseline(TASKS_DIR, max_steps=10)
