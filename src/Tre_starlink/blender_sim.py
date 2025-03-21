import bpy
import socket
import json
import time
import threading
import math
import numpy as np
import os
from datetime import datetime
# from util import release_ports

class BlenderSimulator:
    def __init__(self, blend_file_path, host='127.0.0.1', base_port=5000, num_satellites=12):
        self.blend_file_path = blend_file_path
        self.host = host
        self.base_port = base_port
        self.num_satellites = num_satellites
        self.satellite_ids=['01','02','03','04','05','06','07','08','09','10','11','12']
        self.stop_event = threading.Event()
        self.render_lock = threading.Lock()  # 添加线程锁
        # 加载 Blender 环境文件
        self.load_blender_environment()
        self.satellites,self.cameras,self.sun = self.get_blender_objects()
        pass

    # 加载 Blender 环境文件
    def load_blender_environment(self):
        bpy.ops.wm.open_mainfile(filepath=self.blend_file_path)
        self.scene = bpy.context.scene
        # 设置渲染分辨率
        self.scene.render.resolution_x = 200
        self.scene.render.resolution_y = 200


        # 设置渲染引擎为 Cycles
        self.scene.render.engine = 'CYCLES'

        # 设置使用的设备（CPU 或 GPU）
        self.scene.cycles.device = 'CPU'

        # 设置线程数，这里设置为自动检测
        self.scene.render.threads_mode = 'AUTO'

    # 获取 Blender 场景中的对象
    def get_blender_objects(self):
        satellites=[]
        cameras=[]
        for id in self.satellite_ids:
            satellite =bpy.data.objects.get("satellite"+id)
            satellite.scale=(0.1,0.1,0.1)
            satellites.append(satellite)
            camera = bpy.data.objects.get("Camera"+id)
            camera.data.lens = 50    #相机焦距
            camera.scale=(1,1,0.2)
            cameras.append(camera)

            sun = bpy.data.objects.get("Sun")
        return satellites,cameras,sun

    # 连接到 simulate.py 的 TCP/IP 服务器并获取卫星位置
    def get_satellite_positions(self):
        positions = {}
        attitudes = {}
        for i in range(1, self.num_satellites + 1):
            port = self.base_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.connect((self.host, port))
                    client_socket.sendall(b"GET_STATE")
                    data = client_socket.recv(1024).decode()
                    state = json.loads(data)
                    if port == 5001:
                        sun_position = state['sun_position']
                    positions[i] = state['position']
                    attitudes[i] = state['attitude']
            except Exception as e:
                print(f"Error fetching position for satellite {i}: {e}")
        return positions,attitudes,sun_position

    # 更新 Blender 中的卫星位置
    def update_blender_positions(self, positions, attitudes, satellites, sun_position):
        self.sun.rotation_euler = sun_position
        for sat_id, position in positions.items():
            satellite_name = f"satellite_{sat_id}"
            if satellites:
                x, y, z = position
                rx,ry,rz = attitudes[sat_id]
                unit = 1000.0
                # print(satellites[sat_id-1].location)
                satellites[sat_id-1].location = (x / unit, y / unit, z / unit)
                satellites[sat_id-1].rotation_euler = (math.radians(rx), math.radians(ry), math.radians(rz))
                # print('update position:%s,[%f,%f,%f]'%(position,rx,ry,rz))
            else:
                print(f"Satellite object '{satellite_name}' not found in Blender scene.")

    # 主循环
    def update_positions_threads(self):
        try:
            # 持续更新卫星位置
            while not self.stop_event.is_set():
                # 获取卫星位置
                satellite_positions,satellite_attitudes,sun_position = self.get_satellite_positions()
                # 更新 Blender 中的卫星位置
                self.update_blender_positions(satellite_positions, satellite_attitudes, self.satellites,sun_position)
                self.stop_event.wait(0.1)

        except KeyboardInterrupt:
            print("Thread interrupted by Ctrl+C")
            self.stop_event.set()

    # 控制卫星相机拍照
    def take_satellite_photo(self, satellite_id, output_path):
        print(f"satellite {satellite_id} start photo")
        camera=self.cameras[satellite_id-1]
        if camera:
            with self.render_lock:  # 使用线程锁
                self.scene.camera = camera
                self.scene.render.filepath = output_path
                bpy.ops.render.render(write_still=True)
            print(f"Photo taken by satellite {satellite_id} and saved to {output_path}")
        else:
            print(f"Camera object '{camera_name}' not found in Blender scene.")

    # 卫星相机线程
    def camera_thread(self, satellite_id):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.base_port + satellite_id-1000))
        server_socket.listen(1)
        print(f"Camera {satellite_id} listening on port {self.base_port + satellite_id-1000}")
        while not self.stop_event.is_set():
            client_socket, addr = server_socket.accept()
            with client_socket:
                data = client_socket.recv(1024).decode().split(' ')
                if "TAKE_PHOTO" in data:
                    task_id = int(data[1])
                    time_format = "%Y-%m-%d %H:%M:%S.%f"
                    photo_time = datetime.fromtimestamp(float(data[2])).strftime(time_format)[:-3]
                    output_path = WORKSPACE + f'/dataset/task/{task_id}/'
                    if not os.path.exists(output_path):
                        # 如果路径不存在，则创建文件夹
                        os.makedirs(output_path)
                    output_path = output_path + photo_time + ' satellite id:' +str(satellite_id) +'.png'
                    self.take_satellite_photo(satellite_id, output_path)
                    client_socket.sendall(b"PHOTO_TAKEN")
            client_socket.close()
    # 启动卫星相机线程
    def start_camera_threads(self):
        for i in range(12):
            thread = threading.Thread(target=self.camera_thread, args=(i+1,))
            thread.start()

    def show_point(self,points_array):
        # 定义 12 种不同的颜色
        colors = [
            (1, 0, 0, 1),  # 红色
            (0, 1, 0, 1),  # 绿色
            (0, 0, 1, 1),  # 蓝色
            (1, 1, 0, 1),  # 黄色
            (1, 0, 1, 1),  # 紫色
            (0, 1, 1, 1),  # 青色
            (0.5, 0, 0, 1),  # 暗红色
            (0, 0.5, 0, 1),  # 暗绿色
            (0, 0, 0.5, 1),  # 暗蓝色
            (0.5, 0.5, 0, 1),  # 暗黄色
            (0.5, 0, 0.5, 1),  # 暗紫色
            (0, 0.5, 0.5, 1)   # 暗青色
        ]

        # 循环处理每一组点
        for i in range(points_array.shape[0]):
            # 创建一个新的点云对象
            mesh = bpy.data.meshes.new(name=f"PointCloud_{i}")
            obj = bpy.data.objects.new(f"PointCloud_{i}", mesh)
            bpy.context.collection.objects.link(obj)

            # 获取当前组的点
            points = points_array[i]

            # 创建顶点
            vertices = []
            for point in points:
                vertices.append(tuple(point))

            # 创建网格数据
            mesh.from_pydata(vertices, [], [])
            mesh.update()

            # 创建材质
            material = bpy.data.materials.new(name=f"Material_{i}")
            material.use_nodes = True
            nodes = material.node_tree.nodes
            links = material.node_tree.links

            # 清除默认节点
            for node in nodes:
                if node.name != "Material Output":
                    nodes.remove(node)

            # 创建颜色节点
            color_node = nodes.new(type='ShaderNodeBsdfDiffuse')
            color_node.inputs[0].default_value = colors[i]

            # 连接节点
            output_node = nodes.get("Material Output")
            links.new(color_node.outputs[0], output_node.inputs[0])

            # 将材质分配给对象
            obj.data.materials.append(material)

global WORKSPACE
WORKSPACE = '/home/yuyuan/Project/knowledge_graph/Tre_starlink'
blender_file_path = "/home/yuyuan/blender/地球.blend"
simulator = BlenderSimulator(blender_file_path)     #不能放到线程里，程序会崩

# points_array = np.random.rand(12, 10000, 3)
# simulator.show_point(points_array)

simulator.start_camera_threads()


show_env = os.getenv('SHOW', 'true')
show = show_env.lower() == 'true'
if show:
    #显示
    thread = threading.Thread(target=simulator.update_positions_threads)
    thread.start()
else:
    #不显示,后台显示时需要阻塞用
    simulator.update_positions_threads()    
