import math
import json
from datetime import datetime, timedelta
from functools import lru_cache
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from pyproj import Transformer
from skyfield.api import load, EarthSatellite, wgs84, utc
from skyfield.framelib import itrs


#############################
# 卫星类（带轨迹缓存）
#############################
class Satellite:
    def __init__(self, name, tle_line1, tle_line2):
        self.name = name
        self.ts = load.timescale()
        self.satellite = EarthSatellite(tle_line1, tle_line2, name, ts = self.ts)
        self.trajectory_cache = {}  # datetime -> (lat_rad, lon_rad, alt_km)
        self._velocity_cache = {}

    def get_state(self, observation_time):
        if observation_time not in self.trajectory_cache:
            t = self.ts.from_datetime(observation_time)
            geocentric = self.satellite.at(t)
            subpoint = wgs84.subpoint(geocentric)
            self.trajectory_cache[observation_time] = (subpoint.latitude.radians,
                                                       subpoint.longitude.radians,
                                                       subpoint.elevation.km)
        return self.trajectory_cache[observation_time]

    @lru_cache(maxsize = 1000)
    def get_state_with_velocity(self, observation_time):
        t = self.ts.from_datetime(observation_time)
        geocentric = self.satellite.at(t)
        pos_itrs, vel_itrs = geocentric.frame_xyz_and_velocity(itrs)
        velocity_ecef = vel_itrs.km_per_s
        subpoint = wgs84.subpoint(geocentric)
        lat_rad = subpoint.latitude.radians
        lon_rad = subpoint.longitude.radians
        alt_km = subpoint.elevation.km
        east = [-math.sin(lon_rad), math.cos(lon_rad), 0]
        north = [-math.sin(lat_rad) * math.cos(lon_rad),
                 -math.sin(lat_rad) * math.sin(lon_rad),
                 math.cos(lat_rad)]
        velocity_east = sum(v * e for v, e in zip(velocity_ecef, east))
        velocity_north = sum(v * n for v, n in zip(velocity_ecef, north))
        self._velocity_cache[observation_time] = (velocity_east, velocity_north)
        return lat_rad, lon_rad, alt_km, velocity_east, velocity_north


#############################
# 任务规划类
#############################
class MissionPlanner:
    def __init__(self, Config):
        self.config = Config
        self.ts = load.timescale()
        self.geometry_cache = {}

    def generate_empty_models(self, target_polygon):
        """
        将目标区域分割为网格，每个单元为正方形。
        网格边长取 (√2/2)*swath_km ≈ 7.07 km，转换为度：cell_size_deg = 7.07/111.32
        返回列表，每个元素 {'center': (lon, lat), 'cell': Polygon}
        """
        swath = self.config['swath_km']
        cell_size_km = (math.sqrt(2) / 2) * swath  # 根据卫星幅宽计算格网大小
        cell_size_deg = cell_size_km / 111.32
        minx, miny, maxx, maxy = target_polygon.bounds
        empty_models = []
        y = miny
        while y < maxy:
            x = minx
            while x < maxx:
                cell = Polygon([
                    (x, y),
                    (x + cell_size_deg, y),
                    (x + cell_size_deg, y + cell_size_deg),
                    (x, y + cell_size_deg)
                ])
                if cell.intersects(target_polygon):
                    center = (x + cell_size_deg / 2, y + cell_size_deg / 2)
                    center_point = Point(center)

                    # 仅保留中心点在目标区域内的网格
                    if target_polygon.contains(center_point):
                        empty_models.append({'center': center, 'cell': cell})
                x += cell_size_deg
            y += cell_size_deg

        return empty_models

    def compute_satellite_trajectory(self, satellite, start_time, end_time, dt_sec = 60):
        """
        使用实际 TLE 数据计算星下点轨迹，采样间隔 dt_sec（秒）
        返回 list of (lat, lon, time) —— 坐标顺序 (lat, lon, time)
        """
        trajectory = []
        current_time = start_time
        while current_time <= end_time:
            lat_rad, lon_rad, _ = satellite.get_state(current_time)
            lat_deg = math.degrees(lat_rad)
            lon_deg = math.degrees(lon_rad)
            trajectory.append((lat_deg, lon_deg, current_time))
            current_time += timedelta(seconds = dt_sec)

        return trajectory

    def compute_trajectory_coverage(self, trajectory, target_lon, target_lat, max_slew_deg):
        """
        对于相邻采样点，依据星下点与轨迹切线方向构造覆盖条带多边形，
        使用公式：
           effective_width = sat_altitude * tan(max_slew_deg) + (nominal_swath_km / cos(max_slew_deg))**2
        采用投影法计算每个轨迹段的多边形，再求其并集作为总体覆盖区域。
        """
        proj_str = f"+proj=tmerc +lat_0={target_lat} +lon_0={target_lon} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
        transformer_to_proj = Transformer.from_crs("epsg:4326", proj_str, always_xy = True)
        transformer_to_geo = Transformer.from_crs(proj_str, "epsg:4326", always_xy = True)

        # 有效宽度（m）
        sat_alt = self.config['sat_altitude'] * 1000  # 转为 m
        max_slew_rad = math.radians(max_slew_deg)
        half_width_m = sat_alt * math.tan(max_slew_rad)

        swath_polys = []
        for i in range(len(trajectory) - 1):
            lat1, lon1, t1 = trajectory[i]
            lat2, lon2, t2 = trajectory[i + 1]
            # 投影转换
            x1, y1 = transformer_to_proj.transform(lon1, lat1)
            x2, y2 = transformer_to_proj.transform(lon2, lat2)
            # 轨迹切线方向（平面计算）
            dx = x2 - x1
            dy = y2 - y1
            seg_length = math.hypot(dx, dy)
            if seg_length < 1e-6:
                continue
            # 切向单位向量
            ux, uy = dx / seg_length, dy / seg_length
            # 垂直方向 (右侧) 为 (-uy, ux)
            nx, ny = -uy, ux

            # 计算两个端点偏移后在投影系中的坐标
            precision = 6  # 保留小数点后6位
            p1_left = (round(x1 + nx * half_width_m, precision), round(y1 + ny * half_width_m, precision))
            p1_right = (round(x1 - nx * half_width_m, precision), round(y1 - ny * half_width_m, precision))
            p2_left = (round(x2 + nx * half_width_m, precision), round(y2 + ny * half_width_m, precision))
            p2_right = (round(x2 - nx * half_width_m, precision), round(y2 - ny * half_width_m, precision))

            # 显式闭合多边形
            poly_points = [p1_left, p2_left, p2_right, p1_right, p1_left]
            try:
                poly_proj = Polygon(poly_points)
            except Exception as e:
                print(f"投影坐标系多边形创建失败: {e}")
                continue

            # 转回地理坐标
            poly_geo_coords = [transformer_to_geo.transform(x, y) for x, y in poly_proj.exterior.coords]
            poly_geo = Polygon(poly_geo_coords)

            if poly_geo.is_valid:
                swath_polys.append(poly_geo)

        if swath_polys:
            merged_polygon = unary_union(swath_polys)

            return merged_polygon
        else:
            return None

    def haversine_distance(self, lat1_deg, lon1_deg, lat2_deg, lon2_deg):
        """计算两点（单位：度）的球面距离（km）"""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1_deg, lon1_deg, lat2_deg, lon2_deg])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return self.config['earth_radius_km'] * c

    def get_bounding_square(self, center, radius_km):
        """
        计算给定地理坐标中心点和半径的圆的外切正方形的四个顶点坐标。
        参数:
            center (tuple): 中心点的经纬度坐标 (latitude, longitude)
            radius_km (float): 半径，单位为公里

        返回:
           外切矩形
        """
        lon, lat = center

        # 将半径转换成纬度的变化量，假设地球是完美的球体
        delta_lat = radius_km / self.config['earth_radius_km'] * (180 / math.pi)
        # 对于经度，需要根据当前纬度进行调整
        delta_lon = delta_lat / math.cos(math.radians(lat))

        # 计算四个顶点
        top_left = (lon - delta_lon, lat + delta_lat)
        top_right = (lon + delta_lon, lat + delta_lat)
        bottom_left = (lon - delta_lon, lat - delta_lat)
        bottom_right = (lon + delta_lon, lat - delta_lat)
        target_polygon = Polygon([top_left, top_right, bottom_right, bottom_left])

        return target_polygon

    def bearing_between_points(self, lat1, lon1, lat2, lon2):
        dLon = lon2 - lon1
        y = math.sin(math.radians(dLon)) * math.cos(math.radians(lat2))
        x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(
            math.radians(lat2)) * math.cos(math.radians(dLon))
        return (math.degrees(math.atan2(y, x)) + 360) % 360

    def calculate_tilt_signed(self, satellite, observation_time, target_lat_deg, target_lon_deg):
        """
        计算带符号侧摆角：
          - 获取当前星下点及速度向量求局部航向
          - 计算当前点与目标点大圆距离 d_total
          - diff = (目标方位 - 局部航向)，取 diff 的绝对值的正弦乘以 d_total 得 d_cross
          - 侧摆角 = arctan(d_cross / sat_altitude)，符号与 diff 一致
        """
        lat_sat_rad, lon_sat_rad, alt_sat, _, _ = satellite.get_state_with_velocity(observation_time)
        curr_lat_deg = math.degrees(lat_sat_rad)
        curr_lon_deg = math.degrees(lon_sat_rad)
        # 计算局部航向：这里直接利用速度向量
        _, _, _, velocity_east, velocity_north = satellite.get_state_with_velocity(observation_time)
        track_bearing = math.degrees(math.atan2(velocity_east, velocity_north)) % 360
        bearing_to_target = self.bearing_between_points(curr_lat_deg, curr_lon_deg, target_lat_deg, target_lon_deg)
        diff = (bearing_to_target - track_bearing + 180) % 360 - 180
        d_total = self.haversine_distance(curr_lat_deg, curr_lon_deg, target_lat_deg, target_lon_deg)
        d_cross = d_total * math.sin(math.radians(abs(diff)))
        tilt_mag = math.degrees(math.atan(d_cross / self.config['sat_altitude']))
        tilt_signed = tilt_mag if diff >= 0 else -tilt_mag

        return tilt_signed

    def find_empty_model_windows(self, satellite, coarse_traj, empty_models, coverage_polygon):
        """
            对每个空模型（网格单元），检查其中心点是否在卫星覆盖区域内，
            若在则生成观测窗口。
        """
        windows = []
        for model in empty_models:
            # 判断中心点是否在覆盖区域内
            center_lon, center_lat = model['center']
            center_point = Point(center_lon, center_lat)
            if not coverage_polygon.contains(center_point):
                continue

            center = model['center']  # (lon, lat)
            # 在粗采样轨迹中查找与网格中心距离最近的粗采样时刻
            best_coarse_time = None
            min_dist = float('inf')
            for lat, lon, t in coarse_traj:
                d = self.haversine_distance(lat, lon, center[1], center[0])
                if d < min_dist:
                    min_dist = d
                    best_coarse_time = t
            if best_coarse_time is None:
                continue
            # 在 best_coarse_time 周围进行精细采样：例如 ±30秒内，1秒采样
            fine_traj = []
            t_fine = best_coarse_time - timedelta(seconds = 30)
            while t_fine <= best_coarse_time + timedelta(seconds = 30):
                lat_rad, lon_rad, _ = satellite.get_state(t_fine)
                fine_traj.append((math.degrees(lat_rad), math.degrees(lon_rad), t_fine))
                t_fine += timedelta(seconds = 1)
            # 在精细采样中选取与网格中心最近的时刻
            best_time = None
            min_fine_dist = float('inf')
            for lat, lon, t in fine_traj:
                d = self.haversine_distance(lat, lon, center[1], center[0])
                if d < min_fine_dist:
                    min_fine_dist = d
                    best_time = t
            if best_time:
                try:
                    tilt = self.calculate_tilt_signed(satellite, best_time, center[1], center[0])
                except Exception as e:
                    print("侧摆角计算错误:", e)
                    tilt = 0

                # 新增：检查侧摆角是否在允许范围内
                if abs(tilt) <= self.config['max_side_look_deg']:
                    windows.append({
                        'satellite': satellite.name,
                        'window_start': best_time,
                        'window_end': best_time,
                        'side_look_angle': round(tilt, 2)  # 保留两位小数
                    })
                # 否则不添加到窗口列表

        return windows

    def plan_area_coverage(self, satellites, area_polygon, target_lon, target_lat, start_time, end_time):
        """
        多卫星区域覆盖规划：
          1. 生成目标区域网格（空模型）。
          2. 计算每颗卫星轨迹和覆盖区域（采用条带覆盖方法）。
          3. 对与覆盖区域有交集的空模型，选取轨迹中最近采样时刻作为窗口，并计算侧摆角。
          4. 合并同一卫星中时间接近且网格中心接近、侧摆角差异小的窗口，保留同一时刻只输出一个任务，
             取侧摆角绝对值最小的作为最终结果。
        """
        empty_models = self.generate_empty_models(area_polygon)
        final_plan = []
        for sat in satellites:
            trajectory = self.compute_satellite_trajectory(sat, start_time, end_time, dt_sec = 60)
            coverage_polygon = self.compute_trajectory_coverage(trajectory, target_lon, target_lat,
                                                                self.config['max_side_look_deg'])
            windows = self.find_empty_model_windows(sat, trajectory, empty_models, coverage_polygon)
            final_plan.extend(windows)

        # 1、卫星在同一时间内仅能对一个目标进行观测
        # 2、观测最小时间间隔进行限制
        # 3、云层覆盖率约束主要用于对观测区域的含云量进行约束
        # 4、观测总时长约束主要用于对卫星最终要执行观测目标的总体观测时间进行约束
        # 约束后再输出覆盖的观测窗口
        #

        return final_plan

    def process_tasks(self, tasks_json, satellites):
        results = {}
        for task_id, task in tasks_json.items():
            start_time = datetime.strptime(task["Validity_period"][0], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo = utc)
            end_time = datetime.strptime(task["Validity_period"][1], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo = utc)
            # 目标区域输入格式为 [lon, lat]
            task = tasks_json["0"]
            target_lon, target_lat = task["LLA"][:2]
            # 根据输入中心点经纬度和半径，计算外切矩形四个顶点坐标
            r_km = task["Radius"]
            target_polygon = self.get_bounding_square((target_lon, target_lat), r_km)

            plan = self.plan_area_coverage(satellites, target_polygon, target_lon, target_lat, start_time, end_time)

            formatted_plan = []
            for rec in plan:
                formatted_plan.append({
                    "satellite": rec["satellite"],
                    "window": {
                        "start": rec["window_start"].strftime("%Y-%m-%d %H:%M:%S"),
                        "end": rec["window_end"].strftime("%Y-%m-%d %H:%M:%S")
                    },
                    "side_look_angle": round(rec["side_look_angle"], 2)
                })
            results[task["location"]] = formatted_plan
        return results


#############################
# 主程序入口
#############################
if __name__ == "__main__":
    tasks_json = {
        "0": {
            "location": "test",
            "LLA": [120.1533, 30.2484],
            "Radius": 50,
            "Validity_period": ["2025-03-8 6:20:0.00000", "2025-03-8 8:00:0.00000"],
            "Observation_mode": "cover"
        }
    }
    config = {
        'earth_radius_km': 6371,
        'max_side_look_deg': 30,  # 最大侧摆角 30°
        'sat_altitude': 500,  # 卫星高度 500 km
        'swath_km': 10  # 不侧摆时幅宽 10 km
    }
    tle_json = {"SAT-03":
                    ('1 99999U          24330.93462963  .00000215  00000-0  10961-4 0 00001',
                     '2 99999 097.3992 296.7393 0013211 261.9647 283.6961 15.16828644000019'),
                "SAT-04":
                    ('1 99999U          24330.93462963  .00000252  00000-0  12839-4 0 00008',
                     '2 99999 097.3992 296.7393 0013200 261.9045 238.7564 15.16828577000014'),
                "SAT-05":
                    ('1 99999U          24330.93462963  .00000280  00000-0  14259-4 0 00007',
                     '2 99999 097.3992 296.7393 0013195 261.8916 223.7693 15.16828532000011'),
                "SAT-06":
                    ('1 99999U          24330.93462963  .00000306  00000-0  15580-4 0 00004',
                     '2 99999 097.3992 296.7393 0013190 261.8822 208.7788 15.16828496000019'),
                "SAT-07":
                    ('1 99999U          24330.93462963  .00000323  00000-0  16443-4 0 00002',
                     '2 99999 097.3992 296.7393 0013183 261.8781 193.7828 15.16828482000018'),
                "SAT-08":
                    ('1 99999U          24330.93462963  .00000326  00000-0  16572-4 0 00008',
                     '2 99999 097.3992 296.7393 0013176 261.8825 178.7783 15.16828499000010'),
                "SAT-09":
                    ('1 99999U          24330.93462963  .00000257  00000-0  13088-4 0 00000',
                     '2 99999 097.3992 296.7393 0013204 262.0727 343.5883 15.16828558000015'),
                "SAT-10":
                    ('1 99999U          24330.93462963  .00000251  00000-0  12783-4 0 00005',
                     '2 99999 097.3992 296.7393 0013208 262.0504 328.6106 15.16828571000019'),
                "SAT-11":
                    ('1 99999U          24330.93462963  .00000238  00000-0  12125-4 0 00000',
                     '2 99999 097.3992 296.7393 0013211 262.0228 313.6381 15.16828598000014'),
                "SAT-12":
                    ('1 99999U          24330.93462963  .00000225  00000-0  11430-4 0 00004',
                     '2 99999 097.3992 296.7393 0013212 261.9930 298.6678 15.16828626000016')
                }

    ts = load.timescale()
    satellites = [Satellite(name, tle[0], tle[1]) for name, tle in tle_json.items()]
    planner = MissionPlanner(Config = config)

    # 任务规划：对每颗卫星计算覆盖窗口（基于空模型网格与覆盖区域交集），再合并窗口
    results = planner.process_tasks(tasks_json, satellites)

    print(json.dumps(results, indent = 2))
