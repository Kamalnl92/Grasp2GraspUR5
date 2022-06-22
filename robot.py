import time
import os
import numpy as np
import utils
from simulation import vrep
import trimesh
import cv2
from zmqRemoteApi import RemoteAPIClient
from os.path import exists
class Robot(object):
    def __init__(self, stage, goal_object, obj_mesh_dir, num_obj, workspace_limits,
                 is_testing, test_preset_cases, test_preset_file,
                 goal_conditioned, grasp_goal_conditioned):

        self.workspace_limits = workspace_limits
        self.num_obj = num_obj
        self.stage = stage
        self.goal_conditioned = goal_conditioned
        self.grasp_goal_conditioned = grasp_goal_conditioned
        self.goal_object = goal_object

        # Define colors for object meshes (Tableau palette)
        self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                        # [156, 117, 95], # brown close to green in grey scale
                                        [242, 142, 43], # orange
                                        [237.0, 201.0, 72.0], # yellow
                                        [186, 176, 172], # gray
                                        [255.0, 87.0, 89.0], # red
                                        [176, 122, 161], # purpl
                                        [118, 183, 178], # cyan
                                        [255, 157, 167], # pink
                                        [58.0, 100.0, 140.0], # blue2
                                        [140, 107, 70], # brown2
                                        [220, 122, 23], # orange2
                                        [207.0, 160, 52], # yellow2
                                        # To test for 20 objects in the scene more object of the color grey have been included
                                        [166, 156, 160],[166, 156, 160],[166, 156, 160],[166, 156, 160],[166, 156, 160],[166, 156, 160],[166, 156, 160],
                                        [166, 156, 160],[166, 156, 160],[166, 156, 160],[166, 156, 160],[166, 156, 160],[166, 156, 160],[166, 156, 160],
                                        [166, 156, 160],[166, 156, 160],[166, 156, 160]])/255.0 # gray2

        # inserting the goal object colour green
        green_color = [89.0/255.0, 161.0/255.0, 79.0/255.0]
        self.color_space = np.insert(self.color_space, self.goal_object, green_color, axis=0)

        # color thresholds, small tip if you use CV2 to check the channel ranges, the r and b channels are swapped ;)
        self.color_threshold = [[62, 79, 92, 120, 140, 157], #[120, 145, 90, 120, 75, 94],
                [200, 210, 110, 125, 30, 45],[195, 215, 165, 180, 55, 70], [130, 150, 120, 150, 130, 156],
                 [215, 240, 68, 84, 68, 88], [140, 170, 90, 120, 130, 150], [95, 105, 148, 160, 145, 155],
                 [210, 235, 130, 150, 137, 155], [45, 55, 80, 95, 115, 135], [115, 136, 85, 110, 50, 70],
                 [177, 220, 95, 120, 10, 30], [168, 180, 125, 140, 30, 50], [130, 145, 120, 140, 125, 140],
                 [130, 145, 120, 140, 125, 140], [130, 145, 120, 140, 125, 140], [130, 145, 120, 140, 125, 140],
                [130, 145, 120, 140, 125, 140], [130, 145, 120, 140, 125, 140], [130, 145, 120, 140, 125, 140],
                [130, 145, 120, 140, 125, 140], [130, 145, 120, 140, 125, 140], [130, 145, 120, 140, 125, 140],
                [130, 145, 120, 140, 125, 140], [130, 145, 120, 140, 125, 140],[130, 145, 120, 140, 125, 140],
                [130, 145, 120, 140, 125, 140], [130, 145, 120, 140, 125, 140],[130, 145, 120, 140, 125, 140],
                [130, 145, 120, 140, 125, 140], [130, 145, 120, 140, 125, 140]]

        # inserting the goal object colour green thresholds
        self.color_threshold.insert(self.goal_object, [73, 90, 132, 160, 62, 80])
        # Read files in object mesh directory
        self.obj_mesh_dir = obj_mesh_dir
        self.mesh_list = os.listdir(self.obj_mesh_dir)

        # Randomly choose objects to add to scene
        self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
        self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)), :]
        self.taken_positions_x = []
        self.taken_positions_y = []
        self.Locations_orientations8 = np.array([[]])
        # Connect to simulator
        vrep.simxFinish(-1) # Just in case, close all opened connections   # reason for only one vrep opening?????
        self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')
            self.restart_sim()

        #start the simulations
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        #zeroMQ
        print('Program started')
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')
        print('simulation connected')


        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:

        defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)

        self.is_testing = is_testing
        self.test_preset_cases = test_preset_cases
        self.test_preset_file = test_preset_file

        # Setup virtual camera in simulation
        self.setup_sim_camera()

        # If testing, read object meshes and poses from test case file
        if self.is_testing and self.test_preset_cases:
            file = open(self.test_preset_file, 'r')
            file_content = file.readlines()
            self.test_obj_mesh_files = []
            self.test_obj_mesh_colors = []
            self.test_obj_positions = []
            self.test_obj_orientations = []
            for object_idx in range(self.num_obj):
                file_content_curr_object = file_content[object_idx].split()
                self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir,file_content_curr_object[0]))
                self.test_obj_mesh_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
                self.test_obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),float(file_content_curr_object[6])])
                self.test_obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])
            file.close()
            self.obj_mesh_color = np.asarray(self.test_obj_mesh_colors)

        # Add objects to simulation environment
        self.add_objects()


    def setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4,4)
        cam_trans[0:3,3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4,4)
        cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

    # Function to check the value
    def TouchingAnotherLoc(self, lista, val, margin):
        return(any(abs(item-val) <= margin and abs(item+val) >= margin for item in lista))

    def Locations_orientations_saving(self, Locations_orientations8):
        if exists('locationsOrientations.npy'):
            saved_values = np.load('locationsOrientations.npy')
            new = np.insert(saved_values, np.shape(saved_values)[0], Locations_orientations8, axis=0)
            np.save('locationsOrientations.npy', new)
        else:
            np.save('locationsOrientations.npy', Locations_orientations8)

    def add_objects(self):
        np.random.seed()
        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        sim_obj_handles = []
        if self.stage == 'grasp_only':
            obj_number = 1
            self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
            if self.goal_conditioned or self.grasp_goal_conditioned:
                obj_number = len(self.obj_mesh_ind)
        else:
            obj_number = len(self.obj_mesh_ind)

        # 7 random positions from the saved location orientations
        positions = np.random.choice(7, 7, replace=False)   # /home/kamal/Desktop/HPC/Grasp2GraspUR5/locationsOrientations.npy
        scene = np.random.choice(32, 1, replace=False)  # /home/s3675319/grasp2grasp/Grasp2GraspUR5/locationsOrientations.npy
        scenePositions = np.load('/home/s3675319/grasp2grasp/Grasp2GraspUR5/locationsOrientations.npy')
        # flag = 0
        for object_idx in range(obj_number):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
            if self.is_testing and self.test_preset_cases:
                curr_mesh_file = self.test_obj_mesh_files[object_idx]
            curr_shape_name = 'shape_%02d' % object_idx

            # after creating the 32 scenes I found out that objects close under the robot arm position is not possible to be grasped, hence the offset
            pushedXOffset = 0.125
            # if flag == 3:
            #     print("goal")
            #     drop_x = -0.62#scenePositions[scene[0]][positions[object_idx]*3] - pushedXOffset
            #     drop_y = -0.2#scenePositions[scene[0]][(positions[object_idx]*3)+1]
            #     drop_yaw = scenePositions[scene[0]][(positions[object_idx]*3)+2]
            #     flag += 1
            # else:
            #     print("other")
            drop_x = scenePositions[scene[0]][positions[object_idx]*3] - pushedXOffset
            drop_y = scenePositions[scene[0]][(positions[object_idx]*3)+1]
            drop_yaw = scenePositions[scene[0]][(positions[object_idx]*3)+2]
                # flag += 1

                                                #0.05
            object_position = [drop_x, drop_y, -0.4]#-0.65

            # object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            object_orientation = [0.0, 0.0, drop_yaw] #2*np.pi*np.random.random_sample() # np.random.uniform(-3.14, 3.14)



            if self.is_testing and self.test_preset_cases:
                object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1] + 0.1, self.test_obj_positions[object_idx][2]]
                # object_orientation = [self.test_obj_orientations[object_idx][0], self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
                object_orientation = [0.0, self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]

            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
            try:
                if ret_resp == 8:
                    print('Failed to add new objects to simulation. Please restart.')
                    exit()
                curr_shape_handle = ret_ints[0]
                self.object_handles.append(curr_shape_handle)
                if not (self.is_testing and self.test_preset_cases):
                    time.sleep(0.5)
            except:
                print("curr_shape_handle out of range problem")
                self.restart_sim()
                self.add_objects()


        self.prev_obj_positions = []
        self.obj_positions = []
        np.random.seed(1234)
        # time.sleep(3)


    # def add_objects(self):
    #     np.random.seed()
    #     # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
    #     self.object_handles = []
    #     sim_obj_handles = []
    #     if self.stage == 'grasp_only':
    #         obj_number = 1
    #         self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
    #         if self.goal_conditioned or self.grasp_goal_conditioned:
    #             obj_number = len(self.obj_mesh_ind)
    #     else:
    #         obj_number = len(self.obj_mesh_ind)
    #
    #     drop_yaw = np.random.uniform(-3.14, 3.14)
    #     for object_idx in range(obj_number):
    #         curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
    #         if self.is_testing and self.test_preset_cases:
    #             curr_mesh_file = self.test_obj_mesh_files[object_idx]
    #         curr_shape_name = 'shape_%02d' % object_idx
    #         # drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
    #         # drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1  # + 0.1
    #         taken_position = True
    #         while taken_position:                                                                           #-0.5,1.5
    #             drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.uniform(-0.2,1.3) + self.workspace_limits[0][0] + 0.1 #np.random.uniform(-0.5,0.9)
    #             drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.uniform(-0.2,1.3) + self.workspace_limits[1][0] + 0.1
    #             print("x", drop_x)
    #             print("y", drop_y)
    #                                                                             #0.05
    #             if (not self.TouchingAnotherLoc(self.taken_positions_x, drop_x, 0.045) and
    #                    not self.TouchingAnotherLoc(self.taken_positions_y, drop_y, 0.045)) or not self.taken_positions_x:
    #                 self.taken_positions_x.append(drop_x)
    #                 self.taken_positions_y.append(drop_y)
    #
    #                 # drop_x = np.load('locationsOrientations.npy')[0][0]
    #                 # drop_y = np.load('locationsOrientations.npy')[0][1]
    #
    #                 object_position = [drop_x, drop_y, -0.4]#-0.65
    #                 self.Locations_orientations8 = np.insert(self.Locations_orientations8, np.shape(self.Locations_orientations8)[1], drop_x, axis=1)
    #                 self.Locations_orientations8 = np.insert(self.Locations_orientations8, np.shape(self.Locations_orientations8)[1], drop_y, axis=1)
    #                 self.Locations_orientations8 = np.insert(self.Locations_orientations8, np.shape(self.Locations_orientations8)[1], drop_yaw, axis=1)
    #                 taken_position = False
    #
    #             # object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
    #             object_orientation = [0.0, 0.0, drop_yaw] #2*np.pi*np.random.random_sample() # np.random.uniform(-3.14, 3.14)
    #
    #
    #
    #         if self.is_testing and self.test_preset_cases:
    #             object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1] + 0.1, self.test_obj_positions[object_idx][2]]
    #             # object_orientation = [self.test_obj_orientations[object_idx][0], self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
    #             object_orientation = [0.0, self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
    #         object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]
    #
    #         ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
    #         try:
    #             if ret_resp == 8:
    #                 print('Failed to add new objects to simulation. Please restart.')
    #                 exit()
    #             curr_shape_handle = ret_ints[0]
    #             self.object_handles.append(curr_shape_handle)
    #             if not (self.is_testing and self.test_preset_cases):
    #                 time.sleep(0.5)
    #         except:
    #             print("curr_shape_handle out of range problem")
    #             self.restart_sim()
    #             self.add_objects()
    #
    #
    #     self.prev_obj_positions = []
    #     self.obj_positions = []
    #     print(self.taken_positions_x)
    #     np.random.seed(1234)
    #
    #     print("Type y or n depending scene validation")
    #     validity = input()
    #     print(validity)
    #     if validity == "y":
    #         self.Locations_orientations_saving(self.Locations_orientations8)
    #         print("Locations_orientations8 saved")
    #     elif validity == "n":
    #         print("Locations_orientations8 NOT saved")
    #     else:
    #         print("input not recognised")
    #     assert()


    def restart_sim(self):
        # robot arm1
        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target#0',vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5,0,1.1), vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip#0', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        while gripper_position[2] > 2.0: # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)

        # robot arm2
        sim_ret_cc, self.UR5_target_handle_cc = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle_cc, -1, (-0.5,0,1.1), vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret_cc, self.RG2_tip_handle_cc = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret_cc, gripper_position_cc = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle_cc, -1, vrep.simx_opmode_blocking)

        while gripper_position_cc[2] > 2.0: # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret_cc, gripper_position_cc = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        self.taken_positions_x = []
        self.taken_positions_y = []

    def check_sim(self):

        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.1 and gripper_position[0] < self.workspace_limits[0][1] + 0.1 and gripper_position[1] > self.workspace_limits[1][0] - 0.1 and gripper_position[1] < self.workspace_limits[1][1] + 0.1 and gripper_position[2] > self.workspace_limits[2][0] and gripper_position[2] < self.workspace_limits[2][1]
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()


    def get_task_score(self):

        key_positions = np.asarray([[-0.625, 0.125, 0.0], # red
                                    [-0.625, -0.125, 0.0], # blue
                                    [-0.375, 0.125, 0.0], # green
                                    [-0.375, -0.125, 0.0]]) #yellow

        obj_positions = np.asarray(self.get_obj_positions())
        obj_positions.shape = (1, obj_positions.shape[0], obj_positions.shape[1])
        obj_positions = np.tile(obj_positions, (key_positions.shape[0], 1, 1))

        key_positions.shape = (key_positions.shape[0], 1, key_positions.shape[1])
        key_positions = np.tile(key_positions, (1 ,obj_positions.shape[1] ,1))

        key_dist = np.sqrt(np.sum(np.power(obj_positions - key_positions, 2), axis=2))
        key_nn_idx = np.argmin(key_dist, axis=0)

        return np.sum(key_nn_idx == np.asarray(range(self.num_obj)) % 4)


    def check_goal_reached(self):

        goal_reached = self.get_task_score() == self.num_obj
        return goal_reached


    def get_obj_positions(self):

        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions


    def get_obj_positions_and_orientations(self):

        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)
            obj_orientations.append(object_orientation)

        return obj_positions, obj_orientations

    def mask(self, img, ob_id):
        # rgb thresholds
        # l for lower
        # u for upper
        threshold = self.color_threshold[ob_id]
        maskCh1 = np.where((img[:, :, 0] > threshold[0]) & (img[:, :, 0] < threshold[1]), 1, 0)
        maskCh2 = np.where((img[:, :, 1] > threshold[2]) & (img[:, :, 1] < threshold[3]), 1, 0)
        maskCh3 = np.where((img[:, :, 2] > threshold[4]) & (img[:, :, 2] < threshold[5]), 1, 0)
        mask = np.multiply(maskCh1, maskCh2)
        mask = np.multiply(mask, maskCh3)
        mask = mask*255
        return mask


    def mask_all_obj(self, img, goal_object):
        # threshold = self.color_threshold[goal_object]
        # maskCh1 = np.where((img[:, :, 0] > threshold[0]) & (img[:, :, 0] < threshold[1]), 0, 1)
        # maskCh2 = np.where((img[:, :, 1] > threshold[2]) & (img[:, :, 1] < threshold[3]), 0, 1)
        # maskCh3 = np.where((img[:, :, 2] > threshold[4]) & (img[:, :, 2] < threshold[5]), 0, 1)
        # mask = np.multiply(maskCh1, maskCh2)
        # mask = np.multiply(mask, maskCh3)
        # mask = mask*255

        # cv2.imshow("masks", np.float32(mask))
        # cv2.waitKey(0)
        # cv2.detroyAllWindows()
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        masks = np.full_like(img_grey, 255)
        masks[img_grey <= 55] = 0 # for all the colors
        masks[img_grey == 109] = 0 # for the green color
        masks[img_grey == 101] = 0 # for the green color
        masks[img_grey == 100] = 0 # for the green color
        masks[img_grey == 99] = 0 # for the green color
        masks[img_grey == 97] = 0 # for the green color
        masks[img_grey == 98] = 0 # for the green color
        masks[img_grey == 96] = 0 # for the green color
        masks[img_grey == 94] = 0 # for the green color
        masks[img_grey == 87] = 0 # for the green color
        cv2.imshow("img", img)
        cv2.imshow("img_grey", img_grey)
        cv2.imshow("masks", masks)
        cv2.waitKey(0)
        cv2.detroyAllWindows()
        return masks

    # def obj_contours(self):
    #     obj_contours = []
    #     obj_number = len(self.test_obj_mesh_files)
    #     for object_idx in range(obj_number):
    #         # Get object pose in simulation
    #         sim_ret, obj_position = vrep.simxGetObjectPosition(self.sim_client, self.object_handles[object_idx], -1, vrep.simx_opmode_blocking)
    #         sim_ret, obj_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.object_handles[object_idx], -1, vrep.simx_opmode_blocking)
    #         obj_trans = np.eye(4,4)
    #         obj_trans[0:3,3] = np.asarray(obj_position)
    #         obj_orientation = [obj_orientation[0], obj_orientation[1], obj_orientation[2]]
    #
    #         obj_rotm = np.eye(4,4)
    #         obj_rotm[0:3,0:3] = utils.obj_euler2rotm(obj_orientation)
    #         obj_pose = np.dot(obj_trans, obj_rotm) # Compute rigid transformation representating camera pose
    #         # load .obj files
    #         obj_mesh_file = self.test_obj_mesh_files[object_idx]
    #         # print(obj_mesh_file)
    #
    #         mesh = trimesh.load_mesh(obj_mesh_file)
    #
    #         if obj_mesh_file.split('/')[-1] == '2.obj' or obj_mesh_file.split('/')[-1] == '6.obj':
    #             mesh.apply_transform(obj_pose)
    #         else:
    #             # rest
    #             transformation = np.array([[0,0,1,0],[0,-1,0,0],[1,0,0,0],[0,0,0,1]])
    #             mesh.apply_transform(transformation)
    #             mesh.apply_transform(obj_pose)
    #
    #         obj_contours.append(mesh.vertices[:, 0:2])

        # return obj_contours

    # def obj_contour(self, obj_ind):
        # maxAttemptsToGetPosition = 3
        # for attemp in range(0, maxAttemptsToGetPosition):
        #     try:
        #         # Get object pose in simulation
        #         sim_ret, obj_position = vrep.simxGetObjectPosition(self.sim_client, self.object_handles[obj_ind], -1, vrep.simx_opmode_blocking)
        #         sim_ret, obj_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.object_handles[obj_ind], -1, vrep.simx_opmode_blocking)
        #         break
        #     except:
        #         time.sleep(3)
        #         print("Failed to Get handle camera and Get camera pose and intrinsics in simulation from Coppelia, remaining attempts times in get_obj_mask Function", maxAttemptsToGetPosition-attemp)
        #
        # obj_trans = np.eye(4,4)
        # obj_trans[0:3,3] = np.asarray(obj_position)
        # obj_orientation = [obj_orientation[0], obj_orientation[1], obj_orientation[2]]
        #
        # obj_rotm = np.eye(4,4)
        # obj_rotm[0:3,0:3] = utils.obj_euler2rotm(obj_orientation)
        # obj_pose = np.dot(obj_trans, obj_rotm) # Compute rigid transformation representating camera pose
        #
        # # load .obj files
        # obj_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[obj_ind]])
        # mesh = trimesh.load_mesh(obj_mesh_file)
        #
        # # transform the mesh to world frame
        # if obj_mesh_file.split('/')[-1] == '2.obj' or obj_mesh_file.split('/')[-1] == '6.obj':
        #     mesh.apply_transform(obj_pose)
        # else:
        #     # rest
        #     transformation = np.array([[0,0,1,0],[0,-1,0,0],[1,0,0,0],[0,0,0,1]])
        #     mesh.apply_transform(transformation)
        #     mesh.apply_transform(obj_pose)
        #
        # obj_contour = mesh.vertices[:, 0:2]

        # return obj_contour

    def reposition_objects(self, workspace_limits):

        # Move gripper out of the way
        self.move_to([-0.1, 0, 0.3], None, cc = False)

        for object_handle in self.object_handles:

            # Drop object at random x,y location and random orientation in robot workspace
            drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + workspace_limits[0][0] + 0.1
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.05]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation, vrep.simx_opmode_blocking)
            time.sleep(2)


    def get_camera_data(self):
        maxAttemptsToGetPosition = 4
        for attemp in range(0, maxAttemptsToGetPosition):
            try:
                # Get color image from simulation
                sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
                color_img = np.asarray(raw_image)
                color_img.shape = (resolution[1], resolution[0], 3)
                color_img = color_img.astype(np.float)/255
                color_img[color_img < 0] += 1
                color_img *= 255
                color_img = np.fliplr(color_img)
                color_img = color_img.astype(np.uint8)

                # Get depth image from simulation
                sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
                depth_img = np.asarray(depth_buffer)
                depth_img.shape = (resolution[1], resolution[0])
                depth_img = np.fliplr(depth_img)
                zNear = 0.01
                zFar = 10
                depth_img = depth_img * (zFar - zNear) + zNear
                break
            except:
                time.sleep(10)
                print("Failed to Get get_camera_data", maxAttemptsToGetPosition-attemp)
                self.restart_sim()
                self.add_objects()
                self.get_camera_data()

        return color_img, depth_img

    def close_gripper(self, asynch=False):

        gripper_motor_velocity = -0.5
        gripper_motor_force = 100
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint#0', vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
        gripper_fully_closed = False
        while gripper_joint_position > -0.045: # Block until gripper is fully closed
            sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            if new_gripper_joint_position >= gripper_joint_position:
                return gripper_fully_closed
            gripper_joint_position = new_gripper_joint_position
        gripper_fully_closed = True

        return gripper_fully_closed


    def open_gripper(self, asynch=False):

        gripper_motor_velocity = 0.5
        gripper_motor_force = 20
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint#0', vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
        counter = 0
        while gripper_joint_position < 0.03: # Block until gripper is fully open
            print("gripper_joint_position", gripper_joint_position)
            print(gripper_joint_position)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            counter+=1
            print(counter)
            if counter == 100:
                print("girpper couldn't open restarting simulation")
                counter = 0
                self.restart_sim()
                self.add_objects()



    def move_to(self, tool_position, tool_orientation, cc):

        if cc:
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle_cc, -1, vrep.simx_opmode_blocking)

            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_magnitude/0.02))

            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle_cc,-1,(UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client,self.UR5_target_handle_cc,-1,vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle_cc,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
        else:
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)

            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_magnitude/0.02))

            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client,self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)

    def check_collision(self):
        # listAll = ['/Shape[0]', '/Shape[1]', '/Shape[2]', '/Shape[3]', '/Shape[4]', '/Shape[5]', '/Shape[6]', '/Shape[7]', '/Shape[8]', '/Shape[9]',
        #  '/Shape[10]', '/Shape[11]', '/Shape[12]', '/Shape[13]', '/Shape[14]', '/Shape[15]', '/Shape[16]', '/Shape[17]', '/Shape[18]',
        #  '/Shape[19]', '/Shape[20]']
        listAll = ['/shape_00', '/shape_01', '/shape_02', '/shape_03', '/shape_04', '/shape_05', '/shape_06', '/shape_07', '/shape_08', '/shape_09',
         '/shape_10', '/shape_11', '/shape_12', '/shape_13', '/shape_14', '/shape_15', '/shape_16', '/shape_17', '/shape_18',
         '/shape_19', '/shape_20']
        listSpawned = listAll.copy()
        listSpawned = listSpawned[0:self.num_obj]
        self.sim.callScriptFunction('checkCollision', self.sim.scripttype_mainscript, listSpawned)

    # def stop_check_collision(self):
    #     self.sim.callScriptFunction('noCheckCollision', self.sim.scripttype_mainscript)



    # Primitives ----------------------------------------------------------
    def grasp(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))
        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2
        # Avoid collision with floor
        position = np.asarray(position).copy()
        position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02) #position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02) 0.09
        # Move gripper to location above grasp target
        grasp_location_margin = 0.7 #0.7
        location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)
        # Compute gripper position and linear movement increments
        tool_position = location_above_grasp_target
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.05*move_direction/move_magnitude
        if move_step[0] != 0:
            num_move_steps = int(np.floor(move_direction[0]/move_step[0]))
        else:
            num_move_steps = 0

       # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
        # print("gripper_orientation", gripper_orientation) #        [1.5707963705062866, -0.3926992118358612, 1.5707963705062866]
        # gripper_orientation = [-97.031, 44.65, 4.9182]
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3

        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))
        self.check_collision()
        for step_iter in range(max(num_move_steps, num_rotation_steps)):

            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle_cc,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle_cc, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle_cc,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle_cc, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)
        print("open gripper")
        self.open_gripper()
        self.move_to(position, None, cc = True)
        self.move_to(location_above_grasp_target, None, cc = True)
        # self.close_gripper('RG2_openCloseJoint')
        getCollisionState = self.sim.callScriptFunction('getCollisionState', self.sim.scripttype_mainscript)
        print("colliding", getCollisionState)
        # clearing the collision detection flag in coppeliasim
        self.sim.callScriptFunction('clearCollisionFlag', self.sim.scripttype_mainscript)

        if getCollisionState == 1:
            grasp_success = False
            return grasp_success, None, None, None, None, None

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

        # Ensure gripper is open
        # self.open_gripper(GripperName = 'RG2_openCloseJoint#0')
        # Approach grasp target

        self.move_to(position, None, cc = False)
        # Get images before grasping
        color_img, depth_img = self.get_camera_data()
        depth_img = depth_img * self.cam_depth_scale # Apply depth scale from calibration
        # Get heightmaps before grasping
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, self.cam_intrinsics,
                                                                self.cam_pose, workspace_limits,
                                                                0.002)  # heightmap resolution from args
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
        # Close gripper to grasp target
        gripper_full_closed = self.close_gripper()
        # Move gripper to location above grasp target
        cc = False
        self.move_to(location_above_grasp_target, None, cc = False)
        # Check if grasp is successful
        gripper_full_closed = self.close_gripper()
        grasp_success = not gripper_full_closed

        # Move the grasped object elsewhere
        if grasp_success:
            object_positions = np.asarray(self.get_obj_positions())
            object_positions = object_positions[:,2]
            grasped_object_ind = np.argmax(object_positions)
            print('grasp obj z position', max(object_positions))
            grasped_object_handle = self.object_handles[grasped_object_ind]
            vrep.simxSetObjectPosition(self.sim_client,grasped_object_handle,-1,(-0.5, 0.5 + 0.05*float(grasped_object_ind), 0.1),vrep.simx_opmode_blocking)

            return grasp_success, color_img, depth_img, color_heightmap, valid_depth_heightmap, grasped_object_ind
        else:
            return grasp_success, None, None, None, None, None


    def grasp_non_goal_obj(self, position, heightmap_rotation_angle, workspace_limits):

        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))
        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2
        # Avoid collision with floor
        position = np.asarray(position).copy()
        position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02) #position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02) 0.09
        # Move gripper to location above grasp target
        grasp_location_margin = 0.7 #0.7
        location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)
        # Compute gripper position and linear movement increments
        tool_position = location_above_grasp_target
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.05*move_direction/move_magnitude
        if move_step[0] != 0:
            num_move_steps = int(np.floor(move_direction[0]/move_step[0]))
        else:
            num_move_steps = 0

       # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
        # print("gripper_orientation", gripper_orientation) #        [1.5707963705062866, -0.3926992118358612, 1.5707963705062866]
        # gripper_orientation = [-97.031, 44.65, 4.9182]
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3

        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))
        self.check_collision()
        for step_iter in range(max(num_move_steps, num_rotation_steps)):

            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle_cc,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle_cc, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle_cc,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle_cc, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

        self.open_gripper()
        self.move_to(position, None, cc = True)
        self.move_to(location_above_grasp_target, None, cc = True)
        # self.close_gripper('RG2_openCloseJoint')
        getCollisionState = self.sim.callScriptFunction('getCollisionState', self.sim.scripttype_mainscript)
        print("colliding", getCollisionState)
        # clearing the collision detection flag in coppeliasim
        self.sim.callScriptFunction('clearCollisionFlag', self.sim.scripttype_mainscript)

        if getCollisionState == 1:
            grasp_success = False
            return grasp_success

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

        # Ensure gripper is open
        # self.open_gripper(GripperName = 'RG2_openCloseJoint#0')
        # Approach grasp target

        self.move_to(position, None, cc = False)
        # Get images before grasping
        color_img, depth_img = self.get_camera_data()
        depth_img = depth_img * self.cam_depth_scale # Apply depth scale from calibration
        # Get heightmaps before grasping
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, self.cam_intrinsics,
                                                                self.cam_pose, workspace_limits,
                                                                0.002)  # heightmap resolution from args
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
        # Close gripper to grasp target
        gripper_full_closed = self.close_gripper()
        # Move gripper to location above grasp target
        cc = False
        self.move_to(location_above_grasp_target, None, cc = False)
        # Check if grasp is successful
        gripper_full_closed = self.close_gripper()
        grasp_success = not gripper_full_closed

        # Move the grasped object elsewhere
        if grasp_success:
            object_positions = np.asarray(self.get_obj_positions())
            object_positions = object_positions[:,2]
            grasped_object_ind = np.argmax(object_positions)
            print('grasp obj z position', max(object_positions))
            grasped_object_handle = self.object_handles[grasped_object_ind]
            vrep.simxSetObjectPosition(self.sim_client,grasped_object_handle,-1,(-0.5, 0.5 + 0.05*float(grasped_object_ind), 0.1),vrep.simx_opmode_blocking)

            return grasp_success
        else:
            return grasp_success


