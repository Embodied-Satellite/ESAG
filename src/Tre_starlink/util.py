import astropy.units as u
from astropy.coordinates import EarthLocation, get_sun, AltAz
from astropy.time import Time
import json
import numpy as np
import open3d as o3d
import datetime
import random
import socket
import subprocess
import threading
import re
from scipy.spatial.transform import Rotation as R

def LLA2ECEF(longitude,latitude,height):
    # 创建 EarthLocation 对象
    location = EarthLocation.from_geodetic(lon=longitude, lat=latitude, height=height)

    # 获取 ECEF 坐标
    l = location.to_geocentric()
    return [l[0].value,l[1].value,l[2].value]

def LLA2ECEF_batch(positions_LLA):
    # 创建 EarthLocation 对象
    positions_ECEF=[]
    for position in positions_LLA:
        location = EarthLocation.from_geodetic(lon=position[0], lat=position[1], height=position[2])
        # 获取 ECEF 坐标
        l = location.to_geocentric()
        location_ECEF = [l[0].value,l[1].value,l[2].value]
        positions_ECEF.append(location_ECEF)
    positions_ECEF = np.array(positions_ECEF).astype('float32')/1000     #单位km
    return positions_ECEF

def satellite_ecef_to_ground_ecef(positions):
    """
    将多个卫星的 ECEF 坐标转换为地面对应点的 ECEF 坐标
    :param positions: 一个形状为 (n, 3) 的 numpy 数组，表示 n 个卫星的 ECEF 坐标
    :return: 一个形状为 (n, 3) 的 numpy 数组，表示对应地面点的 ECEF 坐标
    """
    # 计算每个卫星到地心的距离
    satellite_distances = np.sqrt(np.sum(np.square(positions), axis=1))

    # 定义地球半径
    EARTH_RADIUS = 6371

    # 计算每个卫星的缩放比例
    scale_factors = EARTH_RADIUS / satellite_distances

    # 扩展缩放比例以匹配 positions 的形状
    scale_factors = scale_factors[:, np.newaxis]

    # 计算地面点的 ECEF 坐标
    ground_ecef = positions * scale_factors

    return ground_ecef

def display_3d_points_1(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])

    # 创建一个半径为 100 的球体
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=6370)

    # 计算球体的顶点法线，使球体在渲染时更平滑
    sphere.compute_vertex_normals()

    # 设置球体的颜色（这里设置为蓝色）
    sphere.paint_uniform_color([1, 1, 1])

    o3d.visualization.draw_geometries([pcd,coordinate_frame,sphere])

def display_3d_points(points):
    """
    使用 Open3D 显示三维点云
    :param points: 一个形状为 (12, 100, 3) 的 numpy 数组，表示 12 组点集，每组包含 100 个三维点
    """


    # 创建一个 Open3D 可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 定义12种颜色
    colors = [
        [1, 0, 0],   # 红色
        [0, 1, 0],   # 绿色
        [0, 0, 1],   # 蓝色
        [0, 1, 1],   # 青色
        [1, 0, 1],   # 洋红色
        [1, 1, 0],   # 黄色
        [0, 0, 0],   # 黑色
        [1, 0.5, 0], # 橙色
        [0.5, 0, 0.5], # 紫色
        [0.65, 0.16, 0.16], # 棕色
        [1, 0.75, 0.8], # 粉色
        [0.5, 0.5, 0.5]  # 灰色
    ]

    for i in range(12):
        point_set = points[i]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_set)
        pcd.paint_uniform_color(colors[i % len(colors)])
        vis.add_geometry(pcd)

    # 运行可视化
    vis.run()
    vis.destroy_window()



def calculate_angles_between_point_sets(point_set1, point_set2):
    """
    计算卫星观测侧摆角
    """
    # 确保输入是 numpy 数组
    point_set1 = np.array(point_set1)
    point_set2 = np.array(point_set2)
    
    # 计算点积
    dot_products = np.sum(point_set1 * point_set2, axis=1)
    
    # 计算向量的模长
    norms1 = np.linalg.norm(point_set1, axis=1)
    norms2 = np.linalg.norm(point_set2, axis=1)
    
    # 计算夹角的余弦值
    cos_angles = dot_products / (norms1 * norms2)
    
    # 计算夹角（弧度制）
    angles = np.arccos(cos_angles)

    # 计算叉积
    cross_product = np.cross(point_set1, point_set2,axis=1)
    # 确定旋转方向
    normal = np.array([0, 0, 1])
    sign = np.sign(np.dot(cross_product, normal))
    # 计算带符号的夹角
    angles = sign * angles
    
    return angles


def calculate_solar_angles(timestamp,points):
    """
    计算多个观测点的太阳高度角和方位角
    
    参数:
    points (numpy.ndarray): 形状为 (n,4) 的数组，每行格式为 [纬度, 经度, 海拔(m), 时间字符串]
    
    返回:
    tuple: (高度角数组, 方位角数组)，单位为度
    """
    # 分解输入数组
    lons = points[0] * u.deg
    lats = points[1] * u.deg
    heights = points[2]/1000 * u.km
    times = Time(timestamp, format='unix')  # 支持批量时间解析
    
    # 创建观测位置集合
    locations = EarthLocation.from_geodetic(lat=lats, lon=lons, height=heights)     #需要Latitude angle(s) must be within -90 deg <= angle <= 90 deg


    # 向量化计算太阳位置
    sun_positions = get_sun(times)
    
    # 转换到每个位置的地平坐标系
    altaz = sun_positions.transform_to(AltAz(obstime=times, location=locations))
    
    # 提取结果
    elevations = altaz.alt.degree    #高度角
    azimuths = altaz.az.degree    #方位角
    
    return elevations


def calculate_xyz_rotation(x_target, y_target, z_target):
    initial_vector = np.array([0, 0, 1])
    target_vector = np.array([x_target, y_target, z_target])
    target_vector = target_vector / np.linalg.norm(target_vector)
    # 计算旋转向量​
    rotation_vector = np.cross(initial_vector, target_vector)
    rotation_angle = np.arccos(np.dot(initial_vector, target_vector))
    # 使用scipy的Rotation构建旋转对象​
    rot = R.from_rotvec(rotation_angle * rotation_vector / np.linalg.norm(rotation_vector))
    # 获取绕XYZ轴的旋转角​
    euler_angles = rot.as_euler('xyz', degrees=False)
    return (euler_angles[0], euler_angles[1], euler_angles[2])

def task_plan_load(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        # 使用 json.load() 从文件中加载 JSON 数据并转换为 Python 对象
        data = json.load(f)
    
    #处理点目标的task
    point_tasks=[]
    for task_id in data:
        point_task = data[task_id]
        if point_task["location_type"] == "point":
            times=point_task["Validity_period"]
            timestamps=[]
            for time in times:
                # 将时间字符串解析为 datetime 对象
                time_format = "%Y-%m-%d %H:%M:%S.%f"
                dt_obj = datetime.datetime.strptime(time, time_format)
                # 将 datetime 对象转换为时间戳
                timestamp = dt_obj.timestamp()
                timestamps.append(timestamp)

            task=[point_task["LLA"][0],point_task["LLA"][1],point_task["LLA"][2],point_task["task_priority"],point_task["time_priority"],point_task["Quality_priority"]]
            if point_task["Observation_mode"] == "single":
                task.append(1)
            elif point_task["Observation_mode"] == "continuous":
                task.append(2)
            task = [int(task_id)]+timestamps+task
            point_tasks.append(task)
    
    point_tasks=np.array(point_tasks)
    """ 输出依次是:
    task_id,
    time_start,time_end,
    Longitude,latitude,altitude,
    task_priority,  等级1-5, 各项任务的重要性,5代表重要性很高
    time_priority,  等级1-5 ,5代表时间优先级很高
    Quality_priority, 等级1-5 ,5代表拍摄质量优先级很高
    Observation_mode  观测模式1=单次,2=持续
    """
    return point_tasks

def task_plan_simulation(nums,timestart):
    np.random.seed(1)
    task_id = np.arange(nums)
    time_start = timestart+np.random.random(nums)*24*60*60
    time_end = time_start+(np.random.random(nums)+0.1)*24*60*60
    Longitude = (np.random.random(nums)-0.5)*360
    latitude = (np.random.random(nums)-0.5)*160
    altitude = np.random.random(nums)*360
    task_priority = np.random.randint(1, 6, size=nums)
    time_priority = np.random.randint(1, 6, size=nums)
    Quality_priority = np.random.randint(1, 6, size=nums)
    Observation_mode = np.random.randint(1, 3, size=nums)
    point_tasks = [task_id,time_start,time_end,Longitude,latitude,altitude,task_priority,time_priority,Quality_priority,Observation_mode]
    point_tasks = np.array(point_tasks).T
    """ 输出依次是:
    task_id,
    time_start,time_end,
    Longitude,latitude,altitude,
    task_priority,  等级1-5, 各项任务的重要性,5代表重要性很高
    time_priority,  等级1-5 ,5代表时间优先级很高
    Quality_priority, 等级1-5 ,5代表拍摄质量优先级很高
    Observation_mode  观测模式1=单次,2=持续
    """
    return point_tasks



def cloud_prediction(time,location):
    n=len(time)
    return np.random.randint(1, 7, size=n)

def convert_to_native_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    else:
        return obj

def send_command_via_tcp(command, ip="127.0.0.1", port=5003):
    """
    通过 TCP/IP 协议发送命令到指定地址。
    :param command: 要发送的命令字符串
    :param ip: 目标 IP 地址，默认为 127.0.0.1
    :param port: 目标端口，默认为 5005
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ip, port))
            s.sendall(command.encode('utf-8'))
            print(f"Command sent: {command}")
    except Exception as e:
        print(f"Failed to send command: {e}")

def simulation_cmd():

    # Blender 可执行文件路径（根据实际情况修改）
    blender_path = "/home/yuyuan/blender/blender-3.5.1-linux-x64/blender"

    # 要执行的 Blender 脚本路径（根据实际情况修改）
    script_path = "/home/yuyuan/Project/knowledge_graph/Tre_starlink/blender_sim.py"

    # 构造命令参数列表
    command = [
        blender_path,       # Blender 可执行文件
        "-b",               # 后台模式（无界面）
        "--python",         # 执行 Python 脚本
        script_path         # 脚本路径
    ]

    try:
        # 执行命令并捕获输出
        result = subprocess.run(
            command,
            check=True,  # 若命令失败则抛出异常
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # 打印 Blender 输出
        print("Blender 输出：")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Blender 执行失败，错误信息：")
        print(e.stdout)
    except Exception as e:
        print(f"发生意外错误：{str(e)}")

def simulation_cmd_execute():
    # 创建线程
    thread = threading.Thread(target=simulation_cmd)
    # 启动线程
    thread.start()

#kill进程,释放端口
def release_ports(start_port, end_port):
    for port in range(start_port, end_port + 1):
        try:
            # 查找占用端口的进程
            result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True, text=True)
            output = result.stdout
            if output:
                # 使用正则表达式提取 PID
                match = re.search(r'(\d+)', output)
                if match:
                    pid = match.group(1)
                    # 终止进程
                    subprocess.run(['kill', '-9', pid], check=True)
                    print(f"已释放端口 {port}，终止进程 ID 为 {pid} 的进程。")
                else:
                    print(f"无法找到占用端口 {port} 的进程的 PID。")
            else:
                # print(f"端口 {port} 未被占用。")
                pass
        except subprocess.CalledProcessError as e:
            print(f"在释放端口 {port} 时出错: {e}")
        except Exception as e:
            print(f"发生未知错误: {e}")




if __name__ == "__main__":
    # latitude = 30.2484 * u.deg  # 纬度
    # longitude = 120.1533 * u.deg  # 经度
    # height = 7 * u.km  # 海拔高度
    # l=LLA2ECEF(latitude,longitude,height)
    # print(l)

    # path = '/home/yuyuan/Project/knowledge_graph/Tre_starlink/dataset/task_plan.json'
    # point_tasks = task_plan_load(path)

    # point_tasks = task_plan_simulation(10,1)
    # print (point_tasks)

    # simulation_cmd_execute()


    release_ports(5001, 5012)
    release_ports(4001, 4012)