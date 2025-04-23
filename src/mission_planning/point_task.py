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


class MissionPlanner:
    def __init__(self, Config):
        """
        任务规划器初始化
        """
        self.config = Config
        self.ts = load.timescale()

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

    def compute_trajectory_coverage(self, trajectory, max_slew_deg):
        """
        对于相邻采样点，依据星下点与轨迹切线方向构造覆盖条带多边形，

        采用投影法计算每个轨迹段的多边形，再求其并集作为总体覆盖区域。
        """
        target_lon, target_lat = self.current_target
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

    def bearing_between_points(self, lat1, lon1, lat2, lon2):
        dLon = lon2 - lon1
        y = math.sin(math.radians(dLon)) * math.cos(math.radians(lat2))
        x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(
            math.radians(lat2)) * math.cos(math.radians(dLon))
        return (math.degrees(math.atan2(y, x)) + 360) % 360

    # 辅助函数
    #############################
    def haversine_distance(self, lat1_deg, lon1_deg, lat2_deg, lon2_deg):
        """计算两点（单位：度）的球面距离（km）"""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1_deg, lon1_deg, lat2_deg, lon2_deg])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return self.config['earth_radius_km'] * c

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

    def find_visible_windows(self, satellite, coarse_traj, target, coverage_polygon):
        """
            对每个空模型（网格单元），检查其中心点是否在卫星覆盖区域内，
            若在则生成观测窗口。
        """
        windows = []
        # 判断中心点是否在覆盖区域内
        if not coverage_polygon.contains(Point(target[0], target[1])):
            return None

        # 在粗采样轨迹中查找与网格中心距离最近的粗采样时刻
        best_coarse_time = None
        min_dist = float('inf')
        for lat, lon, t in coarse_traj:
            d = self.haversine_distance(lat, lon, target[1], target[0])
            if d < min_dist:
                min_dist = d
                best_coarse_time = t
        if best_coarse_time is None:
            return None
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
            d = self.haversine_distance(lat, lon, target[1], target[0])
            if d < min_fine_dist:
                min_fine_dist = d
                best_time = t
        if best_time:
            try:
                tilt = self.calculate_tilt_signed(satellite, best_time, target[1], target[0])
            except Exception as e:
                print("侧摆角计算错误:", e)
                tilt = 0
            # 检查侧摆角是否在允许范围内
            if abs(tilt) <= self.config['max_side_look_deg']:
                windows.append({
                    'satellite': satellite.name,
                    'window_start': best_time,
                    'window_end': best_time,
                    'side_look_angle': round(tilt, 2)  # 保留两位小数
                })
            # 否则不添加到窗口列表
        return windows

    def find_continuous_observations(self, satellites, target, start_time, end_time):
        """
        收集所有卫星在给定时段内的可视机会，返回按时刻排序的列表。
        """
        self.current_target = target
        final_plan = []
        for sat in satellites:
            trajectory = self.compute_satellite_trajectory(sat, start_time, end_time, dt_sec = 60)
            coverage_polygon = self.compute_trajectory_coverage(trajectory, self.config['max_side_look_deg'])
            windows = self.find_visible_windows(sat, trajectory, target, coverage_polygon)
            if windows:
                final_plan.extend(windows)

        return final_plan

    def process_tasks(self, tasks_json, satellites):
        """
        任务处理主入口
        :param tasks_json: 任务字典
        :param satellites: Satellite对象列表
        :return: 结果字典（可直接转为JSON）
        """
        results = {}
        for task_id, task in tasks_json.items():
            # 时间解析
            start = datetime.strptime(task['Validity_period'][0], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo = utc)
            end = datetime.strptime(task['Validity_period'][1], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo = utc)

            # 坐标转换（输入为[经度, 纬度]）
            target_lon, target_lat = task['LLA'][:2]

            # 连续观测
            ops = self.find_continuous_observations(
                satellites,
                (target_lon, target_lat),
                start,
                end
            )
            formatted_plan = []
            for op in ops:
                formatted_plan.append({
                    "satellite": op["satellite"],
                    "pass_time": op["window_start"].strftime("%Y-%m-%d %H:%M:%S"),
                    "side_look_angle": round(op["side_look_angle"], 2)
                })
            # 排序
            formatted_plan.sort(key = lambda x: x["pass_time"])
            # 连续观测返回值
            if (task['Observation_mode'] == "continuous"):
                results[task["location"]] = formatted_plan
            # 单次观测返回值
            else:
                results[task["location"]] = formatted_plan[0]
        return results


if __name__ == "__main__":
    # 任务格式json
    tasks_json = {
        "1": {
            "location": "location 1",
            "LLA": [120.1533, 30.2484, 0.007],
            "Validity_period": ["2025-03-08 06:0:0.00000", "2025-03-08 09:0:0.00000"],
            "Observation_mode": "continuous"
        }
    }
    # 配置参数
    config = {
        'earth_radius_km': 6371,
        'max_side_look_deg': 30,  # 最大侧摆角 30°
        'sat_altitude': 500,  # 卫星高度 500 km
        'swath_km': 10  # 不侧摆时幅宽 10 km

    }

    tle_json = {"SAT-01":
                    ('1 99999U          24330.93462963  .00000217  00000-0  11019-4 0 00008',
                     '2 99999 097.3992 296.7393 0013208 261.9403 268.7206 15.16828641000018'),
                "SAT-02":
                    ('1 99999U          24330.93462963  .00000230  00000-0  11687-4 0 00004',
                     '2 99999 097.3992 296.7393 0013204 261.9204 253.7404 15.16829277000013'),
                "SAT-03":
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

    satellites = [
        Satellite(
            name = name,
            tle_line1 = tle[0],
            tle_line2 = tle[1],
        ) for name, tle in tle_json.items()
    ]

    # 创建规划器
    planner = MissionPlanner(Config = config)
    # 执行任务规划
    results = planner.process_tasks(
        tasks_json = tasks_json,
        satellites = satellites
    )

    # 转换为JSON友好格式
    print(json.dumps(results, indent = 2))
