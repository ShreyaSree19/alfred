import cv2
import copy
import gen.constants as constants
import numpy as np
from collections import Counter, OrderedDict
from env.tasks import get_task
from ai2thor.controller import Controller
import gen.utils.image_util as image_util
from gen.utils import game_util
from gen.utils.game_util import get_objects_of_type, get_obj_of_type_closest_to_obj


DEFAULT_RENDER_SETTINGS = {'renderImage': True,
                           'renderDepthImage': False,
                           'renderClassImage': False,
                           'renderObjectImage': False,
                           }

class ThorEnv(Controller):
    '''
    an extension of ai2thor.controller.Controller for ALFRED tasks
    '''
    def __init__(self, x_display=constants.X_DISPLAY,
                 player_screen_height=constants.DETECTION_SCREEN_HEIGHT,
                 player_screen_width=constants.DETECTION_SCREEN_WIDTH,
                 quality='MediumCloseFitShadows',
                 build_path=constants.BUILD_PATH):
        self.task = None
        super().__init__(quality=quality)
        self.local_executable_path = build_path
        print(x_display)
        self.start(x_display=x_display,
                   player_screen_height=player_screen_height,
                   player_screen_width=player_screen_width)
        self.task = None

        # internal states
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

        # intemediate states for CoolObject Subgoal
        self.cooled_reward = False
        self.reopen_reward = False

        self.action_history = []
        self.last_failed_action = None

        print("ThorEnv started.")

    def reset(self, scene_name_or_num,
              grid_size=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
              camera_y=constants.CAMERA_HEIGHT_OFFSET,
              render_image=constants.RENDER_IMAGE,
              render_depth_image=constants.RENDER_DEPTH_IMAGE,
              render_class_image=constants.RENDER_CLASS_IMAGE,
              render_object_image=constants.RENDER_OBJECT_IMAGE,
              visibility_distance=constants.VISIBILITY_DISTANCE):
        '''
        reset scene and task states
        '''
        print("Resetting ThorEnv")

        if type(scene_name_or_num) == str:
            scene_name = scene_name_or_num
        else:
            scene_name = 'FloorPlan%d' % scene_name_or_num
        super().reset(scene_name)
        event = super().step(dict(
            action='Initialize',
            gridSize=grid_size,
            cameraY=camera_y,
            renderImage=render_image,
            renderDepthImage=render_depth_image,
            renderClassImage=render_class_image,
            renderObjectImage=render_object_image,
            visibility_distance=visibility_distance,
            makeAgentsVisible=False,
        ))
        # reset task if specified
        if self.task is not None:
            self.task.reset()

        # clear object state changes
        self.reset_states()

        return event

    def reset_states(self):
        '''
        clear state changes
        '''
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()
    
    # def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
    #     event = super().step(dict(
    #         action='Initialize',
    #         gridSize=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
    #         cameraY=constants.CAMERA_HEIGHT_OFFSET,
    #         renderImage=constants.RENDER_IMAGE,
    #         renderDepthImage=constants.RENDER_DEPTH_IMAGE,
    #         renderClassImage=constants.RENDER_CLASS_IMAGE,
    #         renderObjectImage=constants.RENDER_OBJECT_IMAGE,
    #         visibility_distance=constants.VISIBILITY_DISTANCE,
    #         makeAgentsVisible=False,
    #     ))

    #     if len(object_toggles) > 0:
    #         event = super().step(dict(action='SetObjectToggles', objectToggles=object_toggles))

    #     if dirty_and_empty:
    #         event = super().step(dict(action='SetStateOfAllObjects', StateChange="CanBeDirty", forceAction=True))
    #         event = super().step(dict(action='SetStateOfAllObjects', StateChange="CanBeFilled", forceAction=False))

    #     event = super().step(dict(action='SetObjectPoses', objectPoses=object_poses))

    #     # Dummy step to ensure last_event is valid for PutObject
    #     event = self.step(dict(action='Pass'))

    #     self.last_event = event
    #     return event
    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        '''
        restore object locations and states
        '''
        super().step(dict(
            action='Initialize',
            gridSize=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
            cameraY=constants.CAMERA_HEIGHT_OFFSET,
            renderImage=constants.RENDER_IMAGE,
            renderDepthImage=constants.RENDER_DEPTH_IMAGE,
            renderClassImage=constants.RENDER_CLASS_IMAGE,
            renderObjectImage=constants.RENDER_OBJECT_IMAGE,
            visibility_distance=constants.VISIBILITY_DISTANCE,
            makeAgentsVisible=False,
        ))
        if len(object_toggles) > 0:
            super().step((dict(action='SetObjectToggles', objectToggles=object_toggles)))

        if dirty_and_empty:
            super().step(dict(action='SetStateOfAllObjects',
                               StateChange="CanBeDirty",
                               forceAction=True))
            super().step(dict(action='SetStateOfAllObjects',
                               StateChange="CanBeFilled",
                               forceAction=False))
        super().step((dict(action='SetObjectPoses', objectPoses=object_poses)))


    # def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
    #     '''
    #     restore object locations and states
    #     '''
    #     super().step(dict(
    #         action='Initialize',
    #         gridSize=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
    #         cameraY=constants.CAMERA_HEIGHT_OFFSET,
    #         renderImage=constants.RENDER_IMAGE,
    #         renderDepthImage=constants.RENDER_DEPTH_IMAGE,
    #         renderClassImage=constants.RENDER_CLASS_IMAGE,
    #         renderObjectImage=constants.RENDER_OBJECT_IMAGE,
    #         visibility_distance=constants.VISIBILITY_DISTANCE,
    #         makeAgentsVisible=False,
    #     ))
    #     if len(object_toggles) > 0:
    #         for toggle_info in object_toggles:
                    
    #                 action = 'ToggleObjectOn' if toggle_info['isOn'] else 'ToggleObjectOff'
    #                 self.step(dict(
    #                     action=action,
    #                     objectId=toggle_info['objectId'],
    #                     forceAction=True
    #                 ))
    #     # if dirty_and_empty:
    #     #     super().step(dict(action='SetStateOfAllObjects',
    #     #                        StateChange="CanBeDirty",
    #     #                        forceAction=False))
    #     #     super().step(dict(action='SetStateOfAllObjects',
    #     #                        StateChange="CanBeFilled",
    #     #                        forceAction=False))
    #     # need to teleoprt object
    #     for object_pose in object_poses:
    #         super().step(dict(
    #             action='TeleportObject',
    #             objectId=object_pose["objectId"],
    #             position=object_pose["position"],
    #             rotation=object_pose["rotation"]
    #         ))

    def set_task(self, traj, args, reward_type='sparse', max_episode_length=2000):
        '''
        set the current task type (one of 7 tasks)
        '''
        task_type = traj['task_type']
        self.task = get_task(task_type, traj, self, args, reward_type=reward_type, max_episode_length=max_episode_length)

    def check_post_conditions(self, action):
        '''
        handle special action post-conditions
        '''
        if action['action'] == 'ToggleObjectOn':
            self.check_clean(action['objectId'])

    def update_states(self, action):
        '''
        extra updates to metadata after step
        '''
        # add 'cleaned' to all object that were washed in the sink
        event = self.last_event
        if event.metadata['lastActionSuccess']:
            # clean
            if action['action'] == 'ToggleObjectOn' and "Faucet" in action['objectId']:
                sink_basin = get_obj_of_type_closest_to_obj('SinkBasin', action['objectId'], event.metadata)
                cleaned_object_ids = sink_basin['receptacleObjectIds']
                self.cleaned_objects = self.cleaned_objects | set(cleaned_object_ids) if cleaned_object_ids is not None else set()
            # heat
            if action['action'] == 'ToggleObjectOn' and "Microwave" in action['objectId']:
                microwave = get_objects_of_type('Microwave', event.metadata)[0]
                heated_object_ids = microwave['receptacleObjectIds']
                self.heated_objects = self.heated_objects | set(heated_object_ids) if heated_object_ids is not None else set()
            # cool
            if action['action'] == 'CloseObject' and "Fridge" in action['objectId']:
                fridge = get_objects_of_type('Fridge', event.metadata)[0]
                cooled_object_ids = fridge['receptacleObjectIds']
                self.cooled_objects = self.cooled_objects | set(cooled_object_ids) if cooled_object_ids is not None else set()

        return event

    def get_transition_reward(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for transition_reward")
        else:
            return self.task.transition_reward(self.last_event)

    def get_goal_satisfied(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_satisfied(self.last_event)

    def get_goal_conditions_met(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_conditions_met(self.last_event)

    def get_subgoal_idx(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for subgoal_idx")
        else:
            return self.task.get_subgoal_idx()

    def noop(self):
        '''
        do nothing
        '''
        super().step(dict(action='Pass'))

    def smooth_move_ahead(self, action, render_settings=None):
        '''
        smoother MoveAhead
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        smoothing_factor = constants.RECORD_SMOOTHING_FACTOR
        new_action = copy.deepcopy(action)
        new_action['moveMagnitude'] = constants.AGENT_STEP_SIZE / smoothing_factor

        new_action['renderImage'] = render_settings['renderImage']
        new_action['renderClassImage'] = render_settings['renderClassImage']
        new_action['renderObjectImage'] = render_settings['renderObjectImage']
        new_action['renderDepthImage'] = render_settings['renderDepthImage']

        events = []
        for xx in range(smoothing_factor - 1):
            event = super().step(new_action)
            if event.metadata['lastActionSuccess']:
                events.append(event)

        event = super().step(new_action)
        if event.metadata['lastActionSuccess']:
            events.append(event)
        return events

    def smooth_rotate(self, action, render_settings=None):
        '''
        smoother RotateLeft and RotateRight
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        if action['action'] == 'RotateLeft':
            end_rotation = (start_rotation - 90)
        else:
            end_rotation = (start_rotation + 90)

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                    'tempRenderChange': True,
                    'renderNormalsImage': False,
                    'renderImage': render_settings['renderImage'],
                    'renderClassImage': render_settings['renderClassImage'],
                    'renderObjectImage': render_settings['renderObjectImage'],
                    'renderDepthImage': render_settings['renderDepthImage'],
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def smooth_look(self, action, render_settings=None):
        '''
        smoother LookUp and LookDown
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + constants.AGENT_HORIZON_ADJ * (1 - 2 * int(action['action'] == 'LookUp'))
        position = event.metadata['agent']['position']

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': rotation,
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                    'tempRenderChange': True,
                    'renderNormalsImage': False,
                    'renderImage': render_settings['renderImage'],
                    'renderClassImage': render_settings['renderClassImage'],
                    'renderObjectImage': render_settings['renderObjectImage'],
                    'renderDepthImage': render_settings['renderDepthImage'],
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': rotation,
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def look_angle(self, angle, render_settings=None):
        '''
        look at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + angle
        position = event.metadata['agent']['position']

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': rotation,
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': np.round(end_horizon, 3),
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        event = super().step(teleport_action)
        return event

    def rotate_angle(self, angle, render_settings=None):
        '''
        rotate at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        end_rotation = start_rotation + angle

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': np.round(end_rotation, 3),
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': horizon,
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        event = super().step(teleport_action)
        return event
    
    
    def pos_dict_to_array(self, pos_dict):
        return np.array([pos_dict["x"], pos_dict["y"], pos_dict["z"]])

    def _pos_cost(self, pos_dict, object_location):
        #add distance between robot and object
        pos_score = 0
        robot_angle = pos_dict["rotation"] / 180 * np.pi
        robot_angle = np.pi / 2 - robot_angle
        rob_to_obj_vec = self.pos_dict_to_array(object_location) - self.pos_dict_to_array(pos_dict)
        robot_vec = np.array([np.cos(robot_angle), np.sin(robot_angle)])
        rob_to_obj_2d = np.array([rob_to_obj_vec[0], rob_to_obj_vec[2]])
        rob_to_obj_2d = rob_to_obj_2d / np.linalg.norm(rob_to_obj_2d)
        angle_diff = np.arccos(np.dot(robot_vec, rob_to_obj_2d))
        pos_score += angle_diff * 10
        pos_score += 100*np.linalg.norm(rob_to_obj_vec)
        pos_score += pos_dict["horizon"] * -1 + 60
        pos_score += 0 if pos_dict["standing"] else 100
        return pos_score
    
    def move_to_dict(self, pos_dict, mode="teleport"):
        if mode == "teleport":
            pos_x, pos_y, pos_z, pos_rotation, pos_horizon = pos_dict["x"],  pos_dict["y"],  pos_dict["z"],  float(pos_dict["rotation"]),  float(pos_dict["horizon"])
            event = self.step(dict(action="TeleportFull", x=pos_x, y=pos_y, z=pos_z, rotation=pos_rotation, horizon=pos_horizon))
            return event
        elif mode == "navigate":
            raise NotImplementedError

    # def move_to_obj(self, obj):
    #     obj_pos = obj["position"]
        
    #     # 1. Get reachable positions
    #     event = self.step(dict(action="GetReachablePositions"))
    #     interactable_positions = event.metadata["actionReturn"]
        
    #     if not interactable_positions:
    #         print("Error no interactable positions found")
    #         event.metadata["lastActionSuccess"] = False
    #         return event
            
    #     best_cost = np.inf
    #     best_full_pose = None
    #     ROTATIONS = np.arange(0, 360, 10) 
        
    #     # NOTE: You only search for the best rotation (Yaw) in your current logic.
    #     # The Horizon is usually searched for *after* teleporting, to check visibility.
    #     # We will still store the best Yaw.
        
    #     for pos in interactable_positions:
    #         # Ensure keys exist for safe dict manipulation
    #         if "rotation" not in pos:
    #             pos["rotation"] = 0
    #         if "horizon" not in pos:
    #             pos["horizon"] = 0
                
    #         current_best_rot_cost = np.inf
    #         current_best_rotation = pos["rotation"] 

    #         # 2. Search for the optimal Y-axis rotation (Yaw)
    #         for rotation in ROTATIONS:
    #             temp_pose = pos.copy()
    #             temp_pose["rotation"] = rotation
                
    #             # Assuming self._pos_cost calculates a cost based on distance and angle difference
    #             cost = self._pos_cost(temp_pose, obj_pos) 
                
    #             if cost < current_best_rot_cost:
    #                 current_best_rot_cost = cost
    #                 current_best_rotation = rotation

    #         # 3. Check if this position/rotation combination is the best overall
    #         if current_best_rot_cost < best_cost:
    #             best_cost = current_best_rot_cost
                
    #             # Store the full pose: position (x, y, z) + best rotation (Yaw)
    #             best_full_pose = pos.copy()
    #             best_full_pose["rotation"] = current_best_rotation
    #             # Keep horizon at 0 for initial teleport, unless you optimize for it here
    #             best_full_pose["horizon"] = 0 

    #     print(f"Teleporting to best pose keys: {best_full_pose.keys()}")
        
    #     # 4. Execute the teleport using the updated 'move_to_dict' logic
    #     success = self.move_to_dict(best_full_pose, mode="teleport")
        
    #     # You might want to remove this if you only intend to teleport
    #     # event = self.step("MoveAhead") 
        
    #     # Return the success of the teleport action
    #     return success

    def _print_relative_diagnostics(self, agent_pose, object_location):
        """
        Calculates and prints the distance and angle between the agent's final pose 
        and the object's position.
        """
        
        # Ensure inputs are valid
        if "rotation" not in agent_pose:
            print("ERROR: Agent pose is missing 'rotation' key for diagnostics.")
            return
            
        print("\n--- VISIBILITY DIAGNOSTICS ---")
        
        # --- A. Calculate Distance ---
        rob_to_obj_vec = self.pos_dict_to_array(object_location) - self.pos_dict_to_array(agent_pose)
        distance = np.linalg.norm(rob_to_obj_vec)
        # Provide a simple diagnosis based on the numbers:
        if distance > 1.5:
            print(" Distance is > 1.5m (THOR Visibility Threshold).")

    def move_to_obj(
		self,
		obj,
	):
        obj_pos=obj["position"]
        event = self.step(dict(
				action="GetReachablePositions"),
			)
        interactable_positions = event.metadata["actionReturn"]
        if not interactable_positions:
            print("Error no interactable positions found")
            event.metadata["lastActionSuccess"] = False
            return event
        # interactable positions exist
        best_cost = np.inf
        best_full_pose = None
        ROTATIONS = np.arange(0, 360, 10) 
        HORIZONS = [-60, -30, 0, 30]
        
        for i, pos in enumerate(interactable_positions):
            pos["standing"] = True
            
            if "rotation" not in pos:
                pos["rotation"] = 0.0
            if "horizon" not in pos:
                pos["horizon"] = 0.0
                
            # The base position (x, y, z) is fixed for this inner search
            # not finding the right x,y, z
            current_best_rot_cost = np.inf
            current_best_rotation = pos["rotation"] # Default to the initial rotation

            # 1a. Search for the optimal rotation at the current position (x, y, z)
            for rotation in ROTATIONS:
                # Create a temporary pose for cost calculation
                temp_pose = pos.copy()
                temp_pose["rotation"] = rotation
                
                # Use _pos_cost
                cost = self._pos_cost(temp_pose, obj_pos)
                
                if cost < current_best_rot_cost:
                    current_best_rot_cost = cost
                    current_best_rotation = rotation

            # 1b. Check if this position/rotation combination is the best overall
            if current_best_rot_cost < best_cost:
                best_cost = current_best_rot_cost
                
                # Store the full pose (position + best rotation)
                best_full_pose = pos.copy()
                best_full_pose["rotation"] = current_best_rotation
            # if cost < best_cost:
            #     # print("BEST COST: ", cost)
            #     best_cost = cost
                # PSUEDOCODE
                # Compute closest reachable position to the object
                # Using code from pos cost, compute the rotation with the lowest angle diff (in 10 deg increments) to the object
                # Teleport robot to that position and rotation
                # Check horizons from  -30, 0, 30, -60 to find in which one the object is visible
        best_horizon = 0 # Default to 0
    # Use the position and rotation found in the loop
        
        for horizon in HORIZONS:
            best_full_pose["horizon"] = horizon
            
            # Teleport to this temporary pose to check visibility
            self.move_to_dict(best_full_pose, mode="teleport")
            
            # This requires an actual check against the current frame
            # Assuming self.is_object_visible(obj) exists and uses self.last_event
            if obj["visible"]: 
                best_horizon = horizon
                break # Found a visible horizon, stop searching

        # 3. Teleport robot to the final optimized full pose.
        best_full_pose["horizon"] = best_horizon # Set the final best horizon 
    
        # ---  DIAGNOSTIC PRINTING BLOCK ---
        # print(best_full_pose.keys())
        success = self.move_to_dict(
            best_full_pose, mode="teleport"
        )
        agent_final_pose = success.metadata["agent"] 
        obj_pos = obj["position"] # Object position
        print("Agent pose", agent_final_pose)
        print("obj", obj_pos)
        print("rotation", best_full_pose["rotation"])
        print("horizon", best_full_pose["horizon"])
        # # Calculate and print diagnostics
        # self._print_relative_diagnostics(agent_final_pose, obj_pos)
        
        
        # event = self.step("MoveAhead")
        return success

    #     return success
    def to_thor_api_exec(self, action, object_id="", action_str="", obj=None, smooth_nav=False):
        """
        Convert a high-level action string to a valid THOR API action.
        Adds extensive safety checks and debug logs to locate 'NoneType' errors.
        """
        try:
            # === Movement actions ===
            if "RotateLeft" in action:
                event = self.step(dict(action="RotateLeft", forceAction=False), smooth_nav=smooth_nav)

            elif "RotateRight" in action:
                event = self.step(dict(action="RotateRight", forceAction=False), smooth_nav=smooth_nav)

            elif "MoveAhead" in action:
                event = self.step(dict(action="MoveAhead", forceAction=False), smooth_nav=smooth_nav)

            elif "LookUp" in action:
                event = self.step(dict(action="LookUp", forceAction=False), smooth_nav=smooth_nav)

            elif "LookDown" in action:
                event = self.step(dict(action="LookDown", forceAction=False), smooth_nav=smooth_nav)

            # === Interaction actions ===
            elif "OpenObject" in action:
                if not object_id:
                    print("[WARN] Missing object_id for OpenObject.")
                    return self.last_event, action

                visible_objs = [obj['objectId'] for obj in self.last_event.metadata['objects'] if obj['visible']]
                if object_id not in visible_objs:
                    print(f"[WARN] {object_id} not visible. Skipping OpenObject.")
                    return self.last_event, action

                event = self.step(dict(action="OpenObject", objectId=object_id, moveMagnitude=1.0))

            elif "CloseObject" in action:
                if not object_id:
                    print("[WARN] Missing object_id for CloseObject.")
                    return self.last_event, action

                visible_objs = [obj['objectId'] for obj in self.last_event.metadata['objects'] if obj['visible']]
                if object_id not in visible_objs:
                    print(f"[WARN] {object_id} not visible. Skipping CloseObject.")
                    return self.last_event, action

                event = self.step(dict(action="CloseObject", objectId=object_id, forceAction=False))

            elif "PickupObject" in action:
                if not object_id:
                    print("[WARN] Missing object_id for PickupObject.")
                    return self.last_event, action
                event = self.step(dict(action="PickupObject", objectId=object_id))

            elif "PutObject" in action:
                print("[DEBUG] Entered PutObject branch.")
                if not self.last_event:
                    print("[ERROR] self.last_event is None before PutObject.")
                    return self.last_event, action

                metadata = getattr(self.last_event, "metadata", None)
                if not metadata:
                    print("[ERROR] Missing metadata in last_event.")
                    return self.last_event, action

                inventory_objects = metadata.get("inventoryObjects", [])
                if not inventory_objects:
                    print("[WARN] No inventory objects found in metadata.")
                    return self.last_event, action

                inventory_object_id = inventory_objects[0].get("objectId", None)
                if not inventory_object_id:
                    print("[ERROR] Inventory object missing objectId.")
                    return self.last_event, action

                print(f"[DEBUG] Holding object {inventory_object_id}, placing it on {object_id}")
                event = self.step(dict(
                    action="PutObject",
                    objectId=inventory_object_id,
                    # receptacleObjectId=object_id,
                    forceAction=False,
                    placeStationary=True
                ))

            elif "ToggleObjectOn" in action:
                event = self.step(dict(action="ToggleObjectOn", objectId=object_id))

            elif "ToggleObjectOff" in action:
                event = self.step(dict(action="ToggleObjectOff", objectId=object_id))

            elif "SliceObject" in action:
                if not self.last_event or "inventoryObjects" not in self.last_event.metadata:
                    print("[ERROR] SliceObject attempted but no knife held or metadata missing.")
                    return self.last_event, action

                inventory_objects = self.last_event.metadata.get("inventoryObjects", [])
                if not inventory_objects or "Knife" not in inventory_objects[0].get("objectType", ""):
                    print("[ERROR] Agent not holding a knife before slicing.")
                    return self.last_event, action

                event = self.step(dict(action="SliceObject", objectId=object_id))

            elif "GoToObject" in action:
                if not obj or "objectId" not in obj:
                    print("[ERROR] obj is None for GoToObject.")
                    return self.last_event, action
                print(f"[DEBUG] Moving to object: {obj.get('name', 'unknown')}")
                event = self.move_to_obj(obj=obj)

            else:
                raise Exception(f"Invalid action: {action}")

            # === Result handling ===
            if event is None:
                print(f"[WARN] Event is None after executing action: {action_str}")
            elif "metadata" not in dir(event) or "lastActionSuccess" not in event.metadata:
                print(f"[WARN] Event missing metadata or lastActionSuccess: {action_str}")
            elif event.metadata["lastActionSuccess"]:
                self.last_failed_action = None
                self.action_history.append(action_str)
                print("ACTION SUCCESS:", action_str)
            else:
                self.last_failed_action = action_str
                print(event.metadata["errorMessage"])
                print("ACTION FAILED:", action_str)

            return event, action

        except TimeoutError:
            print(f"[ERROR] Timeout during to_thor_api_exec for action {action_str}. Skipping.")
            return self.last_event, action
        except RuntimeError as e:
            if "Target object not found" in str(e) or "not visible" in str(e):
                print(f"[WARN] Object missing or not visible during {action_str}. Skipping.")
                return self.last_event, action
            else:
                raise e
        except Exception as e:
            print(f"[ERROR] Exception during to_thor_api_exec for action {action_str}: {e}")
            import traceback
            traceback.print_exc()
            return self.last_event, action



    def check_clean(self, object_id):
        '''
        Handle special case when Faucet is toggled on.
        In this case, we need to execute a `CleanAction` in the simulator on every object in the corresponding
        basin. This is to clean everything in the sink rather than just things touching the stream.
        '''
        event = self.last_event
        if event.metadata['lastActionSuccess'] and 'Faucet' in object_id:
            # Need to delay one frame to let `isDirty` update on stream-affected.
            event = self.step({'action': 'Pass'})
            sink_basin_obj = game_util.get_obj_of_type_closest_to_obj("SinkBasin", object_id, event.metadata)
            for in_sink_obj_id in sink_basin_obj['receptacleObjectIds']:
                if (game_util.get_object(in_sink_obj_id, event.metadata)['dirtyable']
                        and game_util.get_object(in_sink_obj_id, event.metadata)['isDirty']):
                    event = self.step({'action': 'CleanObject', 'objectId': in_sink_obj_id})
        return event

    def prune_by_any_interaction(self, instances_ids):
        '''
        ignores any object that is not interactable in anyway
        '''
        pruned_instance_ids = []
        for obj in self.last_event.metadata['objects']:
            obj_id = obj['objectId']
            if obj_id in instances_ids:
                if obj['pickupable'] or obj['receptacle'] or obj['openable'] or obj['toggleable'] or obj['sliceable']:
                    pruned_instance_ids.append(obj_id)

        ordered_instance_ids = [id for id in instances_ids if id in pruned_instance_ids]
        return ordered_instance_ids
    
    def convert_action_to_string(self, obj, act, held_obj):
        if act == "PutObject":
            action = "Put " + held_obj["objectType"] + " in " + obj["objectType"]
        else:
            action = act.replace("Object", " " + obj["objectType"] + " ").lower()
        
        return action

    def generate_possible_actions(self):
        '''
        ignores any object that is not interactable in anyway
        '''

        objects = self.last_event.metadata["objects"]
        held_object = None
        #held object
        inventory_objects = self.last_event.metadata['inventoryObjects']
        if (len(inventory_objects) > 0):
            held_object = inventory_objects[0]
 
        actions = []
        for obj in objects:
            if obj['pickupable'] and held_object is None:
                actions.append({"obj": obj, "action": "PickupObject", "action_str": self.convert_action_to_string(obj, "PickupObject", held_object)})
            if obj['receptacle'] and held_object is not None:
                actions.append({"obj": obj, "action": "PutObject", "action_str": self.convert_action_to_string(obj, "PutObject", held_object)})
            if obj['openable']:
                if obj['isOpen']:
                     actions.append({"obj": obj, "action": "OpenObject", "action_str": self.convert_action_to_string(obj, "OpenObject", held_object)})
                else:
                     actions.append({"obj": obj, "action": "CloseObject", "action_str": self.convert_action_to_string(obj, "CloseObject", held_object)})
            if obj['toggleable']:
                if obj['isToggled']:
                     actions.append({"obj": obj, "action": "ToggleObjectOn", "action_str": self.convert_action_to_string(obj, "ToggleObjectOn", held_object)})
                else:
                     actions.append({"obj": obj, "action": "ToggleObjectOff", "action_str": self.convert_action_to_string(obj, "ToggleObjectOff", held_object)})
            if obj['sliceable'] and held_object is not None and "knife" in held_object['objectType'].lower():
                actions.append({"obj": obj, "action": "SliceObject", "action_str": self.convert_action_to_string(obj, "SliceObject", held_object)})
            actions.append({"obj": obj, "action": "GoToObject", "action_str": self.convert_action_to_string(obj, "GoToObject", held_object)})

        return actions

    def va_interact(self, action, interact_mask=None, smooth_nav=True, mask_px_sample=1, debug=False):
        '''
        interact mask based action call
        '''

        all_ids = []

        if type(interact_mask) is str and interact_mask == "NULL":
            raise Exception("NULL mask.")
        elif interact_mask is not None:
            # ground-truth instance segmentation mask from THOR
            instance_segs = np.array(self.last_event.instance_segmentation_frame)
            color_to_object_id = self.last_event.color_to_object_id

            # get object_id for each 1-pixel in the interact_mask
            nz_rows, nz_cols = np.nonzero(interact_mask)
            instance_counter = Counter()
            for i in range(0, len(nz_rows), mask_px_sample):
                x, y = nz_rows[i], nz_cols[i]
                instance = tuple(instance_segs[x, y])
                instance_counter[instance] += 1
            if debug:
                print("action_box", "instance_counter", instance_counter)

            # iou scores for all instances
            iou_scores = {}
            for color_id, intersection_count in instance_counter.most_common():
                union_count = np.sum(np.logical_or(np.all(instance_segs == color_id, axis=2), interact_mask.astype(bool)))
                iou_scores[color_id] = intersection_count / float(union_count)
            iou_sorted_instance_ids = list(OrderedDict(sorted(iou_scores.items(), key=lambda x: x[1], reverse=True)))

            # get the most common object ids ignoring the object-in-hand
            inv_obj = self.last_event.metadata['inventoryObjects'][0]['objectId'] \
                if len(self.last_event.metadata['inventoryObjects']) > 0 else None
            all_ids = [color_to_object_id[color_id] for color_id in iou_sorted_instance_ids
                       if color_id in color_to_object_id and color_to_object_id[color_id] != inv_obj]

            # print all ids
            if debug:
                print("action_box", "all_ids", all_ids)

            # print instance_ids
            instance_ids = [inst_id for inst_id in all_ids if inst_id is not None]
            if debug:
                print("action_box", "instance_ids", instance_ids)

            # prune invalid instances like floors, walls, etc.
            instance_ids = self.prune_by_any_interaction(instance_ids)

            # cv2 imshows to show image, segmentation mask, interact mask
            if debug:
                print("action_box", "instance_ids", instance_ids)
                instance_seg = copy.copy(instance_segs)
                instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] == 1
                instance_seg *= 255

                cv2.imshow('seg', instance_segs)
                cv2.imshow('mask', instance_seg)
                cv2.imshow('full', self.last_event.frame[:,:,::-1])
                cv2.waitKey(0)

            if len(instance_ids) == 0:
                err = "Bad interact mask. Couldn't locate target object"
                success = False
                return success, None, None, err, None

            target_instance_id = instance_ids[0]
        else:
            target_instance_id = ""

        if debug:
            print("taking action: " + str(action) + " on target_instance_id " + str(target_instance_id))
        try:
            event, api_action = self.to_thor_api_exec(action, target_instance_id, smooth_nav)
        except Exception as err:
            success = False
            return success, None, None, err, None

        if not event.metadata['lastActionSuccess']:
            if interact_mask is not None and debug:
                print("Failed to execute action!", action, target_instance_id)
                print("all_ids inside BBox: " + str(all_ids))
                instance_seg = copy.copy(instance_segs)
                instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] == 1
                cv2.imshow('seg', instance_segs)
                cv2.imshow('mask', instance_seg)
                cv2.imshow('full', self.last_event.frame[:,:,::-1])
                cv2.waitKey(0)
                print(event.metadata['errorMessage'])
            success = False
            return success, event, target_instance_id, event.metadata['errorMessage'], api_action

        success = True
        return success, event, target_instance_id, '', api_action

    @staticmethod
    def bbox_to_mask(bbox):
        return image_util.bbox_to_mask(bbox)

    @staticmethod
    def point_to_mask(point):
        return image_util.point_to_mask(point)

    @staticmethod
    def decompress_mask(compressed_mask):
        return image_util.decompress_mask(compressed_mask)