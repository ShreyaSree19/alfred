import json
import traceback
from alfred_agent import Alfred_Agent
from env.thor_env_modified import ThorEnv


def single_eval(agent, env, max_steps=10):
    actions_taken = []

    for step in range(max_steps):
        # actions = agent.env.generate_possible_actions()
        # for action_dict in actions:
        #     action_str = action_dict["action_str"]
        #     if "goto" in action_str.lower():
        #         print("selected: " + action_str)

        #         obj_id = action_dict["obj"]["objectId"]
        #         action_type = action_dict["action"]
                
        #         pos = action_dict["obj"]["position"]

        #         print(obj_id, action_type, action_str)

        #         # try:
        #         event, thor_action = agent.env.to_thor_api_exec(
        #             action=action_type,
        #             object_id=obj_id,
        #             action_str=action_str,
        #             obj=action_dict["obj"]
        #         )
        #         break


        action_dict = agent.select_action()

        if action_dict["action_str"] == "done":
            print(f"[Step {step}] Agent chose to stop.")
            break

        print(f"[Step {step}] Selected action:", action_dict["action_str"])
        obj_id = action_dict["obj"]["objectId"]
        action_type = action_dict["action"]
        action_str = action_dict["action_str"]
        pos = action_dict["obj"]["position"]

        print(obj_id, action_type, action_str)

        # try:
        event, thor_action = agent.env.to_thor_api_exec(
            action=action_type,
            object_id=obj_id,
            action_str=action_str,
            obj=action_dict["obj"]
        )
        # except Exception as e:
        #     print(f"[Step {step}] Execution failed:", e)
        #     continue

        actions_taken.append(action_dict)

        # Check task success
        # if env.get_goal_satisfied():
        #     print(f"[Step {step}] Goal satisfied!")
        #     return True, actions_taken

    return actions_taken


def run_single_eval(task_json_path, scene_name="FloorPlan1", max_steps=10):
    # Load task JSON
    with open(task_json_path, "r") as f:
        task_dict = json.load(f)

    # Initialize environment and agent
    env = ThorEnv()
    env.reset(scene_name)
    agent = Alfred_Agent(env=env)
    task = "make a bacon, lettuce, and tomato sandwich"
    agent.set_task(task)  # or any appropriate field from your JSON

    # # Optional: Set task graph if available
    # if "task_graph_file" in task_dict:
    #     agent.set_task_graph_path(task_dict["task_graph_file"])

    try:
        print(task_dict.keys())
        # print(f"Running evaluation on task: {task_dict['task_desc']}")
        actions = single_eval(agent, env, max_steps=max_steps)
        print("Executed actions:")
        for a in actions:
            print(f" - {a['action_str']} (p={a['probability']:.2f})")
    except Exception:
        print(traceback.format_exc())
        success, actions = False, []

    env.stop()
    return success, actions


def main():
    task_path = "Tasks/Make_A_BLT_0_abridged.json"  # Modify as needed
    scene_name = "FloorPlan1"
    max_steps = 10

    run_single_eval(task_path, scene_name, max_steps)


if __name__ == "__main__":
    main()