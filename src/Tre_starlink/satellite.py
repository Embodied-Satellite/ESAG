import sys
sys.path.append('/home/mars/cyh_ws/ESAG/') 

import math
import threading
import time
import json
import os
import socket
import datetime
import re
import numpy as np
import logging
from sgp4.api import Satrec
from sgp4.api import jday
from astropy.time import Time
from astropy.coordinates import EarthLocation, ITRS, GCRS, CartesianRepresentation, get_sun
import astropy.units as u
import faiss
from src.Tre_starlink.util import calculate_solar_angles,task_plan_load,LLA2ECEF_batch,cloud_prediction,task_plan_simulation,convert_to_native_types
from src.Tre_starlink.util import LLA2ECEF,display_3d_points,satellite_ecef_to_ground_ecef,display_3d_points_1,calculate_angles_between_point_sets,simulation_cmd_execute,release_ports,calculate_xyz_rotation
from sklearn.cluster import DBSCAN
from src.Tre_starlink.plan import GeneticAlgorithm
import copy
#卫星功能层
class Satellite:
    def __init__(self, id, tle_line1, tle_line2, start_time ,Planning_duration , lock_photo):
        self.id = id
        self.Satrec = Satrec.twoline2rv(tle_line1, tle_line2)
        self.start_time = start_time
        self.last_update_time = self.start_time  # 初始化最后更新时间为起始时间
        self.Planning_duration = Planning_duration

        # 初始姿态设置
        # 相机位于-Z轴,通信位于Y轴
        # Y 轴（翻滚角）平行于轨道，-Z 轴（偏航角）指向地心
        self.attitude = (0, 0, 0)  # 初始姿态，(rx, ry, rz) 分别为俯仰角、翻滚角、偏航角
        self.current_roll = 0  # 当前翻滚角（绕Y轴）
        self.target_attitude = [self.start_time, 0]  # 目标姿态定义为绕Y轴的翻滚角，-Z轴相对指向地心时角度为0
        self.attitude_velocity = 0.5  # 侧摆速度，单位：度/秒

        #线程锁
        self.lock_photo = lock_photo
        self.lock = threading.Lock()
        #卫星连接顺序
        self.task_list = []
        self.connection_order = [9, 10, 11, 12, 3, 1, 2, 4, 5, 6, 7, 8]

        # 打印调试信息以验证姿态变化
        log_file = WORKSPACE + f"/logs/satellite_{self.id}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger = logging.getLogger(f"Satellite_{self.id}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)

    def rotate_vector_around_axis(self, vector, axis, angle):
        """绕指定轴旋转向量"""
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        axis_unit = axis / np.linalg.norm(axis)
        rotated_vector = (
            vector * cos_theta +
            np.cross(axis_unit, vector) * sin_theta +
            axis_unit * np.dot(axis_unit, vector) * (1 - cos_theta)
        )
        return rotated_vector

    def update_position(self, current_time):
        """根据当前时间更新卫星的位置和姿态，并进行坐标转换。
        卫星运动是地球自转与轨道运动的叠加。
        输出相对于静止地球的卫星位置。"""
        # 将当前时间转换为儒略日
        self.current_time = current_time
        jd, fr = jday(*time.gmtime(current_time)[:6])
        e, r, v = self.Satrec.sgp4(jd, fr)
        if e != 0:
            self.logger.error(f"Error calculating position for satellite {self.id}: SGP4 error code {e}")
            return

        # 更新卫星的位置和速度
        # 将位置向量 r 分解为 x, y, z 分量 (ECI坐标系: 地心惯性坐标系)
        x, y, z = r
        # 将速度向量 v 分解为 vx, vy, vz 分量 (ECI坐标系: 地心惯性坐标系)
        vx, vy, vz = v

        # 地球自转角速度 (radians per second)
        earth_rotation_period = 86164.0905  # 地球自转周期 (秒)
        omega_earth = 2 * math.pi / earth_rotation_period  # 地球自转角速度
        # 考虑地球自转对卫星位置和速度的影响
        elapsed_time = current_time - self.start_time
        theta_earth = omega_earth * elapsed_time  # 地球自转角度

        # 将卫星的 ECI 坐标转换为地固坐标系 (ECEF)
        x_rotated = x * math.cos(theta_earth) + y * math.sin(theta_earth)
        y_rotated = -x * math.sin(theta_earth) + y * math.cos(theta_earth)
        z_rotated = z

        # 将速度向量从 ECI 转换到 ECEF
        vx_rotated = vx * math.cos(theta_earth) + vy * math.sin(theta_earth)
        vy_rotated = -vx * math.sin(theta_earth) + vy * math.cos(theta_earth)
        vz_rotated = vz

        # 更新卫星的位置,ECEF（地固坐标系）
        self.position = (x_rotated, y_rotated, z_rotated)
        self.velocity = (vx_rotated, vy_rotated, vz_rotated)

        #ECI（地心惯性坐标系）下,计算卫星姿态
        # 定义目标方向向量
        velocity_vector = np.array(self.velocity)
        velocity_unit = velocity_vector / np.linalg.norm(velocity_vector)  # 归一化速度向量
        position_vector = np.array(self.position)
        earth_center_vector = -position_vector / np.linalg.norm(position_vector)  # 指向地心的单位向量

        # 初始化旋转矩阵
        R = np.zeros((3, 3))
        R[:, 1] = velocity_unit  # 新Y轴对齐速度向量
        R[:, 2] = -earth_center_vector  # 初始状态：新-Z轴指向地心 (翻滚角为0度)

        # 平滑过渡到目标翻滚角
        target_roll = self.target_attitude[1]  # 目标翻滚角
        delta_roll = (target_roll - self.current_roll + 180) % 360 - 180  # 计算最短路径
        max_delta = self.attitude_velocity * (current_time - self.last_update_time)  # 基于上次更新时间

        # 打印调试信息以验证关键变量
        # self.logger.info(f"Target roll: {target_roll}, Current roll: {self.current_roll}, Delta roll: {delta_roll}, Max delta: {max_delta}")

        # 确保 max_delta 和 delta_roll 的符号一致
        signed_max_delta = math.copysign(min(abs(max_delta), abs(delta_roll)), delta_roll)

        # 更新当前翻滚角
        self.current_roll = (self.current_roll + signed_max_delta) % 360

        # 翻滚逻辑：从初始状态 -earth_center_vector 开始，沿 velocity_unit 方向旋转到目标角度
        initial_z_axis = -earth_center_vector
        target_z_axis = self.rotate_vector_around_axis(initial_z_axis, velocity_unit, math.radians(self.current_roll))
        R[:, 2] = target_z_axis
        R[:, 0] = np.cross(R[:, 1], R[:, 2])  # 新X轴由新Y和新Z叉乘得到

        # 确保旋转矩阵正交
        R[:, 0] /= np.linalg.norm(R[:, 0])
        R[:, 1] /= np.linalg.norm(R[:, 1])
        R[:, 2] /= np.linalg.norm(R[:, 2])

        # 更新最后更新时间
        self.last_update_time = current_time

        # 将旋转矩阵转换为XYZ顺序的欧拉角
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6  # 判断是否接近奇异点（当sy接近0时，pitch角接近±90度，导致数值不稳定）

        if not singular:  # 如果不是奇异点，使用标准公式计算欧拉角
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        # 转换为角度并更新卫星姿态（ECEF）
        self.attitude = (
            math.degrees(x),
            math.degrees(y),
            math.degrees(z)
        )

        # 更新输出
        self.output = [current_time , self.position, self.attitude,self.current_roll]
        # print(self.output)


    def get_state(self):
        """获取卫星的当前状态，包括位置和姿态"""
        output = copy.copy(self.output)
        satellite_state = {
                'current_time':output[0],
                'position': output[1],
                'attitude': output[2],
                'roll_angle': output[3],
                'energy_allow': 0.4,
            }
        if self.id == 1:
            sun_position = self.get_sun_position(output[0])
            satellite_state['sun_position'] = sun_position
        return satellite_state
        

    def set_target_attitude(self, angle):
        """设置新的目标姿态角度（绕Y轴的翻滚角）"""
        self.logger.info(f"Satellite {self.id} set target attitude with roll angle {angle}°")
        with self.lock:
            timestamp = self.current_time
            self.target_attitude = [timestamp, angle % 360]

    def get_image(self,task_id):
        """通过TCP/IP控制指定相机拍照"""
        self.logger.info(f"Satellite {self.id} agent get image")
        # 创建TCP/IP连接
        self.lock_photo.acquire()
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('127.0.0.1', 4000+self.id))  # 假设相机服务器运行在本地主机端口6000上
        command = f"TAKE_PHOTO {task_id} {self.current_time}"
        client_socket.sendall(command.encode())
        print('photo start')
        data = client_socket.recv(1024).decode()
        self.logger.info(data)
        client_socket.close()
        time.sleep(0.5)
        print('photo end')
        self.lock_photo.release()
        # 解析返回的数据
        pass

    def task_update(self, task_data):
        """使用新任务更新卫星的任务列表，并通过 TCP/IP 传播给邻居卫星"""
        # Update current satellite's task list
        tasks = task_data.get("task", {})
        self.task_list = tasks.get(str(self.id), [])
        self.logger.info(f"Satellite {self.id} updating tasks: {self.task_list}")

        # Determine propagation direction based on satellite_cur
        satellite_cur = task_data.get("satellite_cur")
        if satellite_cur is not None:
            satellite_cur = int(satellite_cur)

            # Define neighbors based on connection order

            try:
                current_index = self.connection_order.index(self.id)
                left_neighbor = self.connection_order[current_index - 1] if current_index > 0 else None
                right_neighbor = self.connection_order[current_index + 1] if current_index < len(self.connection_order) - 1 else None
            except ValueError:
                left_neighbor = None
                right_neighbor = None

            # Log neighbor information for debugging
            self.logger.info(f"Satellite {self.id}: Left Neighbor = {left_neighbor}, Right Neighbor = {right_neighbor}")

            # Determine which neighbor to propagate to
            if self.id == satellite_cur:
                # Start satellite propagates to both neighbors
                neighbors_to_propagate = [left_neighbor, right_neighbor]
            elif self.connection_order.index(self.id) > self.connection_order.index(satellite_cur):
                # Propagate to left neighbor (clockwise direction)
                neighbors_to_propagate = [right_neighbor]
            else:
                # Propagate to right neighbor (counter-clockwise direction)
                neighbors_to_propagate = [left_neighbor]

            # Remove None values from neighbors_to_propagate
            neighbors_to_propagate = [n for n in neighbors_to_propagate if n is not None]

            for neighbor_id in neighbors_to_propagate:
                try:
                    # Call task_update on neighbor satellite via TCP/IP
                    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client_socket.connect(('127.0.0.1', 5000 + neighbor_id))
                    command = f"TASK_UPDATE {json.dumps(task_data)}"
                    client_socket.sendall(command.encode())
                    client_socket.close()
                    self.logger.info(f"Called task_update on Satellite {neighbor_id}")
                except Exception as e:
                    self.logger.error(f"Failed to call task_update on Satellite {neighbor_id}: {str(e)}")

    def task_manager(self):
        """管理并执行任务列表中的任务"""
        while True:
            if not self.task_list:
                time.sleep(0.1)
                continue

            # Sort tasks by timestamp
            task = self.task_list.pop(0)

            # Check if the first task is due for execution
            task_pattern = r'^\s*T:(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+C:(?P<action>\w+)\s+P:(?P<params>.*)\s*$'
            match = re.match(task_pattern, task)
            if not match:
                self.logger.error("Invalid task format")
                return

            timestamp = match.group('timestamp')
            action = match.group('action')
            params = match.group('params')

            task_time = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f").timestamp()

            while self.current_time < task_time:
                time.sleep(0.005)

            self.task_execution(action, params)

    def task_execution(self, action, params):
        """根据动作和参数执行特定任务"""
        self.logger.info(f"Satellite {self.id} executing task: C:{action} P:{params}")
        try:
            if action == "SET_ATTITUDE":
                # print(raw_angles)
                # raw_angles = params.strip('()').split(',')
                # target_angles = tuple(float(angle.strip()) for angle in raw_angles)
                # self.set_target_attitude(*target_angles)

                self.set_target_attitude(float(params))
            elif action == "GET_IMAGE":
                self.get_image(int(params))
            else:
                self.logger.error("Unknown task command")
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")

    def calculate_positions_24h(self):
        """
        计算卫星在24小时内每隔1分钟的位置点。

        功能描述：
        - 使用SGP4模型高效计算卫星位置。
        - 时间范围：24小时，时间间隔：1分钟（60秒）。
        - 输出格式：[(timestamp, (x, y, z)), ...]，其中timestamp为时间戳，(x, y, z)为卫星位置坐标。

        返回:
            list: 包含时间戳和对应位置坐标的列表。
        """
        positions = []
        start_time = self.start_time
        end_time = start_time + self.Planning_duration  # 24小时
        current_time = start_time

        while current_time <= end_time:
            jd, fr = jday(*time.gmtime(current_time)[:6])
            e, r, v = self.Satrec.sgp4(jd, fr)
            if e != 0:
                raise ValueError(f"SGP4 error code {e} at time {current_time}")
            positions.append((current_time, tuple(r)))
            current_time += 1  # 每隔1s计算一次

        return positions

    def get_sun_position(self,current_time):
        obs_time = Time(current_time, format='unix', scale='utc')

        sun_gcrs = get_sun(obs_time)  #GCRS即ECI
        
        # 将太阳位置转换为 ITRS (ECEF) 坐标系
        sun_itrs = sun_gcrs.transform_to(ITRS(obstime=obs_time)) #ITRS是高精度的ECEF,考虑了地球的变形和板块运动
        
        # 提取 ECEF 坐标 (单位转换为公里)
        x = sun_itrs.x.to(u.km).value
        y = sun_itrs.y.to(u.km).value
        z = sun_itrs.z.to(u.km).value

        # 计算绕XYZ轴旋转的旋转角
        RXYZ = calculate_xyz_rotation(x, y, z)
        return RXYZ

    def Solar_Array_Control(self):
        # 根据卫星姿态和太阳位置,地球位置(遮挡)调整帆板角度,提高充电效率
        # 每隔1分钟调整一次帆板角度
        pass

#通信层
class Satellite_interaction(threading.Thread):
    def __init__(self, satellite, lock_photo, host='127.0.0.1', base_port=5000):
        super().__init__()
        self.satellite = satellite
        self.daemon = True
        self.host = host
        self.port = base_port + satellite.id
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 确保端口可重新使用
        self.stop_event = threading.Event()
        self.lock_photo = lock_photo

        # 配置代理日志记录器
        log_file = WORKSPACE+f"/logs/satellite_{satellite.id}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger = logging.getLogger(f"Agent_{satellite.id}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)

    def run(self):
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.logger.info(f"Satellite {self.satellite.id} agent listening on port {self.port}")

            task_thread = threading.Thread(target=self.satellite.task_manager, daemon=True)
            task_thread.start()
            
            while not self.stop_event.is_set():
                self.server_socket.settimeout(1)
                try:
                    client_socket, addr = self.server_socket.accept()
                    data = client_socket.recv(9012).decode()
                    try:
                        # Handle commands with or without parameters
                        parts = data.strip().split(' ', 1)
                        command = parts[0]
                        parameters = parts[1] if len(parts) > 1 else None
                        # self.lock.acquire()
                        if command == "SET_ATTITUDE":
                            try:
                                # Extract the angle from the command string
                                angle = float(parameters.strip())
                                self.satellite.set_target_attitude(angle)
                                response = f"Set satellite {self.satellite.id} target attitude to {angle}°"
                            except ValueError:
                                response = "Error: Invalid angle format. Use: SET_ATTITUDE <angle>"
                            except Exception as e:
                                response = f"Error: {str(e)}"
                        elif command == "GET_STATE":
                            state = self.satellite.get_state()
                            response = json.dumps(state)
                        elif command == "GET_IMAGE":
                            try:
                                task_id = int(parameters.strip())
                                self.satellite.get_image(task_id)
                                response = f"satellite {self.satellite.id} get image°"
                            except Exception as e:
                                response = f"Error processing GET_IMAGE: {str(e)}"
                        elif command == "TASK_UPDATE":
                            task_data = json.loads(parameters)
                            self.satellite.task_update(task_data)
                            response = f"Updated tasks for satellite {self.satellite.id}"
                        else:
                            response = "Unknown command"
                        client_socket.sendall(response.encode())
                    except ValueError:
                        client_socket.sendall(b"Invalid command format. Use: SET_ATTITUDE <value>")
                    finally:
                        client_socket.close()
                        # self.lock.release()
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            self.logger.info(f"\nSatellite {self.satellite.id} agent stopped")
        finally:
            self.server_socket.close()

# 星链规划层
class SatelliteLinks:
    def __init__(self, num_satellites, start_time, Planning_duration, lock_photo, simulation_acc=1000):
        self.satellites = []
        self.connection_order = [9, 10, 11, 12, 3, 1, 2, 4, 5, 6, 7, 8]  # 自定义连接顺序
        self.start_time = start_time  #time.time()
        self.current_time = self.start_time
        self.Planning_duration = Planning_duration
        self.simulation_acc = simulation_acc  # 仿真加速因子
        self.lock_photo=lock_photo
        # Read TLE data from file
        tle_file_path = WORKSPACE+"/dataset/轨道.tle"
        with open(tle_file_path, 'r') as file:
            lines = file.readlines()

        # Parse TLE data into satellite objects
        for i in range(min(num_satellites, len(lines) // 3)):
            name = lines[i * 4].strip()
            line1 = lines[i * 4 + 1].strip()
            line2 = lines[i * 4 + 2].strip()
            satellite = Satellite(i + 1, line1, line2, self.start_time,Planning_duration,lock_photo)
            self.satellites.append(satellite)

    def update_positions(self):
        """Update positions of all satellites with acceleration, following connection order"""
        while True:
            self.lock_photo.acquire()
            self.current_time = self.current_time + 0.01*self.simulation_acc
            self.lock_photo.release()
            for satellite in self.satellites:
                satellite.update_position(self.current_time)
            time.sleep(0.01)

    def start_simulation(self):
        simulation_thread = threading.Thread(target=self.update_positions, daemon=True)
        simulation_thread.start()

    def get_all_state(self):
        """Get positions of all satellites, following connection order"""
        states = {}
        for satellite in self.satellites:
            states[satellite.id] = satellite.get_state()
        return states

    def print_state(self, interval=0.5):
        """Periodically print the state of all satellites and Earth's attitude, following connection order"""
        while True:
            time.sleep(interval)
            states = self.get_all_state()
            # elapsed_time = self.current_time - self.start_time
            print("\n--- Printing Satellite States ---")
            for sat_id in self.connection_order:
                if sat_id in states:
                    state = states[sat_id]
                    print(f"Satellite {sat_id}:")
                    print(f"  Position: {state[sat_id]['position']}")
                    print(f"  Attitude: {state[sat_id]['attitude']}°")
                    print(f"  Target_attitude: {state[sat_id]['target_attitude']}°")

    def convert_eci_to_ecef(self,positions_eci):
        # 地球自转角速度 (radians per second)
        current_time = positions_eci[:,:,0]
        earth_rotation_period = 86164.0905  # 地球自转周期 (秒)
        omega_earth = 2 * math.pi / earth_rotation_period  # 地球自转角速度
        # 考虑地球自转对卫星位置和速度的影响
        elapsed_time = current_time - self.start_time
        theta_earth = omega_earth * elapsed_time  # 地球自转角度

        # 将卫星的 ECI 坐标转换为地固坐标系 (ECEF)
        x = positions_eci[:,:,1]
        y = positions_eci[:,:,2]
        z = positions_eci[:,:,3]
        x_ECEF = x * np.cos(theta_earth) + y * np.sin(theta_earth)
        y_ECEF = -x * np.sin(theta_earth) + y * np.cos(theta_earth)
        z_ECEF = z
        
        positions_ECEF = np.concatenate((np.expand_dims(current_time,axis=2),np.expand_dims(x_ECEF,axis=2),np.expand_dims(y_ECEF,axis=2),np.expand_dims(z_ECEF,axis=2)),axis=2)
        return positions_ECEF

    # def convert_eci_to_ecef(self,positions_eci):
    #     shape = positions_eci.shape
    #     eci_points = positions_eci.reshape((shape[0]*shape[1],shape[2]))
    #     time = Time(eci_points[:,0], format='unix')

    #     cart_rep = CartesianRepresentation(
    #         x=eci_points[:, 1],
    #         y=eci_points[:, 2],
    #         z=eci_points[:, 3]
    #     )

    #     # 创建 ECI 坐标对象，这里使用 GCRS（近似 ECI）
    #     eci_coords = GCRS(cart_rep, obstime=time)

    #     # 进行坐标转换到 ECEF（用 ITRS 表示 ECEF）
    #     ecef_coords = eci_coords.transform_to(ITRS(obstime=time))

    #     # 提取转换后的 ECEF 坐标
    #     ecef_points = np.column_stack([
    #         eci_points[:,0],
    #         ecef_coords.cartesian.x.value,
    #         ecef_coords.cartesian.y.value,
    #         ecef_coords.cartesian.z.value
    #     ])
    #     ecef_points = ecef_points.reshape(shape)
    #     return ecef_points
    
    def get_satellite_positions_24h(self):
        """
        获取所有卫星在24小时内每隔1分钟的位置点（单线程版本），并将其转换为地面经纬度点。

        功能描述：
        - 使用单线程依次调用每个卫星的`calculate_positions_24h`方法。
        - 将每个位置点从地心惯性坐标系（ECI）转换为地面经纬度点。
        - 输出格式：{sat_id: [(timestamp, (latitude, longitude)), ...], ...}。

        返回:
            dict: 包含每颗卫星ID及其对应地面经纬度点的字典。
        """
        eci_positions = {}
        for satellite in self.satellites:
            eci_positions[satellite.id] = satellite.calculate_positions_24h()
        
        points=[[] for i in range(12)]
        for id in eci_positions:
            eci_position=eci_positions[id]
            for p in eci_position:
                points[id-1].append([p[0],*p[1]])
        positions_ECI=np.array(points)

        positions_ECEF = self.convert_eci_to_ecef(positions_ECI)
        return positions_ECEF

    def get_near_satellite_init(self,positions_ECEF):
        ids = np.array([[[i] for _ in range(positions_ECEF.shape[1])] for i in range(12)])
        positions_ECEF = np.concatenate((ids,positions_ECEF),axis=2)
        self.positions_ECEF_qury = np.reshape(positions_ECEF,(positions_ECEF.shape[0]*positions_ECEF.shape[1],positions_ECEF.shape[2]))
        # 初始化GPU资源
        # self.res = faiss.StandardGpuResources()
        # 创建索引
        self.index = faiss.IndexFlatL2(3)  # 使用L2距离的扁平索引
        # 轨道位置转对应地面位置
        positions_ECEF_ground = satellite_ecef_to_ground_ecef(self.positions_ECEF_qury[:,2:])
        self.positions_ECEF_ground = np.ascontiguousarray(positions_ECEF_ground, dtype=None).astype('float32')
        # 添加数据到索引
        self.index.add(self.positions_ECEF_ground)


    def get_near_satellite(self,position_qury):
        # 定义搜索的距离范围
        radius = 300**2

        # 执行范围搜索
        lims, distances, indices = self.index.range_search(position_qury, radius)
        positions_ECEF_inrange = self.positions_ECEF_qury[indices]
        distances = np.expand_dims(np.sqrt(distances),axis=1)
        positions_ECEF_inrange = np.concatenate((positions_ECEF_inrange ,distances),axis=1)

        # id,timestamp,x,y,z,distance
        return positions_ECEF_inrange


    def get_position_infos(self,positions_ECEF_inrange,task):
        if len(positions_ECEF_inrange)>0:
            # 观测角,太阳角,云层厚度,有效时段,优先级

            # 根据观测卫星和时段,聚类任务
            dbscan = DBSCAN(eps=1, min_samples=1)
            labels = dbscan.fit_predict(positions_ECEF_inrange[:,:2])
            #计算卫星拍摄位置
            photo_positions=[]
            nums=labels.max()
            for i in range(nums+1):
                photo_positions.append(np.median(positions_ECEF_inrange[np.where(labels == i)],axis=0))

            photo_positions = np.array(photo_positions)
            timestamp = photo_positions[:,1]
            Satellite_point = photo_positions[:,2:5]

            # 观测角,侧摆角
            ground_point = np.expand_dims(task[10:],0)
            Observation_angle = calculate_angles_between_point_sets(ground_point-Satellite_point,-Satellite_point)
            # 太阳角
            Solar_angle = calculate_solar_angles(timestamp,task[3:6])
            #云量
            Cloud_thickness = cloud_prediction(timestamp,task[3:5])
            #照片质量
            Observation_angle_Factor = np.clip(-(Observation_angle/3.14*180)**2/900+1,0,None)   #30度以内有效,
            Solar_angle_Factor = np.clip(-(Solar_angle-90)**2/4900+1,0,None)    #20度以内有效
            Cloud_thickness_Factor = np.clip((-Cloud_thickness**2+49)/48,0,None)   #5级以内有效

            image_quality = Observation_angle_Factor*Solar_angle_Factor*Cloud_thickness_Factor

            Observation_angle = np.expand_dims(Observation_angle,axis=1)
            image_quality = np.expand_dims(image_quality,axis=1)
            positions_info = np.concatenate((photo_positions[:,:2],Observation_angle,image_quality),axis=1)

            # 删除质量<=0的值
            indices = np.where(positions_info[:,-1] != 0)[0]
            positions_info = positions_info[indices]
            if positions_info.size == 0 :
                positions_info = None
        else:
            positions_info = None

        
        #输出:卫星id,timestamp,侧摆角,成像质量
        return positions_info

    def tasks_plan(self, tasks, tasks_info, state_info):
        """考虑速度约束下,角度可达
        能源约束(限制和目标优化)
        单次,多次
        任务,时间,质量优先"""
        # 保存tasks_info和state_info到文件
        # with open(WORKSPACE+'/dataset/tasks_state_info.json', 'w') as f:
        #     data = {'tasks':convert_to_native_types(tasks),'tasks_info': convert_to_native_types(tasks_info), 'state_info': convert_to_native_types(state_info)}
        #     json.dump(data, f)
            
        GA = GeneticAlgorithm(starttime = self.start_time,Planning_duration = self.Planning_duration)
        GA.load_tasks(tasks, tasks_info, state_info)
        GA.plan()
        GA.execute()
        task_plan_result = GA.execute()
        return task_plan_result
    
    def run_plan(self, tasks):
        # 获取卫星24小时途径点的ECEF坐标
        positions_ECEF = self.get_satellite_positions_24h()
        # display_3d_points(positions_ECEF[:,:,1:])
        # 任务点的经纬度转ECEF坐标
        point_ECEF = LLA2ECEF_batch(tasks[:, 3:6])
        # 任务=position,观测时间段,优先级,单次/连续,
        tasks = np.concatenate((tasks, point_ECEF), axis=1).astype('float32')

        # 初始化卫星观测窗口搜索器
        self.get_near_satellite_init(positions_ECEF)

        start_time = time.time()
        tasks_info = []
        positions_ECEF_inranges = np.array([[0, 0, 0, 0, 0, 0]])
        for task in tasks:
            position_qury = np.expand_dims(task[10:], 0)
            # 任务点附近卫星搜索
            positions_ECEF_inrange = self.get_near_satellite(position_qury)
            # 卫星路径,显示点云用
            positions_ECEF_inranges = np.concatenate((positions_ECEF_inranges, positions_ECEF_inrange), axis=0)

            # 获取卫星执行拍摄的拍摄质量,过滤不符合要求的卫星
            info = self.get_position_infos(positions_ECEF_inrange, task)
            if info is not None:
                # 记录任务的可执行卫星和相关拍摄信息
                tasks_info.append([task[0], info])  # task id [卫星1观测质量,卫星2观测质量]

        # # #显示任务点和卫星路径点云
        # positions_show = np.concatenate((positions_ECEF_inranges[:,2:5],point_ECEF),axis=0)
        # display_3d_points_1(positions_show)

        if len(tasks_info) > 0:
            # 获取当前时刻卫星状态
            state_info = self.get_all_state()
            # 任务规划和执行
            task_plan_result = self.tasks_plan(tasks, tasks_info, state_info)
            return task_plan_result
        else:
            print('搜索不到拍摄卫星.')
            return None

def satellite_plan_tool():
    # 释放TCP/IP端口5001-5012,多次执行端口可能被占用
    release_ports(5001, 5012)
    release_ports(4001, 4012)

    # 定义线程锁,提高拍摄精度
    global WORKSPACE
    WORKSPACE = '/home/mars/cyh_ws/ESAG/src/Tre_starlink'
    lock_photo = threading.Lock()

    # 定义初始化时间
    # timestamp = '2025-03-07 6:15:14.131254'
    # timestamp = '2025-03-07 6:5:14.131254'
    timestamp = '2025-03-07 5:0:14.131254'
    starttime = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    Planning_duration = 86400 * 2  # 规划时长24h

    # 开启仿真
    num_satellites = 12
    simulation_acc = 1000  # 仿真加速倍数
    satellite_link = SatelliteLinks(num_satellites, start_time=starttime, Planning_duration=Planning_duration, lock_photo=lock_photo, simulation_acc=simulation_acc)
    satellite_link.start_simulation()
    # satellite_link.update_positions()
    # time.sleep(0.1)
    # satellite_link.get_all_state()

    # for satellite in satellite_link.satellites:
    #     satellite.get_sun_position(starttime)

    # 开启通讯
    agents = []
    for satellite in satellite_link.satellites:
        agent = Satellite_interaction(satellite, lock_photo=lock_photo)
        # agent.run()    # 调试用
        agent.start()
        agents.append(agent)

    # 加载观测任务
    path = WORKSPACE + '/dataset/task_plan.json'
    tasks = task_plan_load(path)
    # 随机生成50个任务
    # tasks = task_plan_simulation(50, starttime)

    # 规划+执行task update
    task_plan_result = satellite_link.run_plan(tasks)

    # 返回任务规划结果
    print(f'type(task_plan_result): {type(task_plan_result)}')
    return task_plan_result


if __name__ == "__main__":
    result = satellite_plan_tool()
    print(f'result: {result}')
