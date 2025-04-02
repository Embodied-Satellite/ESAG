import math
import json
import numpy as np
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite, wgs84, utc
from skyfield.framelib import itrs


class Satellite:
    def __init__(self, name, tle_line1, tle_line2, ts = None):
        """
        卫星对象初始化
        :param name: 卫星名称
        :param tle_line1: TLE第一行
        :param tle_line2: TLE第二行
        :param config: 全局配置字典
        :param ts: Skyfield时间尺度对象
        """
        self.name = name
        self.ts = ts or load.timescale()
        self.satellite = EarthSatellite(tle_line1, tle_line2, name = name, ts = self.ts)

    def get_state(self, observation_time):
        """
        返回星下点的经纬度（单位：弧度）和高度（单位：km）
        """
        t = self.ts.from_datetime(observation_time)
        geocentric = self.satellite.at(t)
        subpoint = wgs84.subpoint(geocentric)
        lat_rad = subpoint.latitude.radians
        lon_rad = subpoint.longitude.radians
        altitude_km = subpoint.elevation.km
        return lat_rad, lon_rad, altitude_km

    def get_state_with_velocity(self, observation_time):
        """
        获取卫星状态及速度（向量化计算）
        :return: (纬度_rad, 经度_rad, 高度_km, 东向速度_km/s, 北向速度_km/s)
        """
        t = self.ts.from_datetime(observation_time)
        geocentric = self.satellite.at(t)
        pos_itrs, vel_itrs = geocentric.frame_xyz_and_velocity(itrs)
        velocity_ecef = vel_itrs.km_per_s

        subpoint = wgs84.subpoint(geocentric)
        lat_rad = subpoint.latitude.radians
        lon_rad = subpoint.longitude.radians
        altitude_km = subpoint.elevation.km

        # ENU速度分解
        east = [-math.sin(lon_rad), math.cos(lon_rad), 0]
        north = [
            -math.sin(lat_rad) * math.cos(lon_rad),
            -math.sin(lat_rad) * math.sin(lon_rad),
            math.cos(lat_rad)
        ]
        velocity_east = sum(v * e for v, e in zip(velocity_ecef, east))
        velocity_north = sum(v * n for v, n in zip(velocity_ecef, north))

        return lat_rad, lon_rad, altitude_km, velocity_east, velocity_north


class MissionPlanner:
    def __init__(self):
        """
        任务规划器初始化
        """
        self.config = {
            'earth_radius_km': 6371,
            'max_side_look_deg': 30,
            'fov_deg': 1.1459,
            'min_elevation_deg': 20.0
        }
        self.ts = load.timescale()

    def great_circle_distance_km(self, lat1, lon1, lat2, lon2):
        """
        计算两点(纬度/经度，单位：弧度)在球面上的大圆距离(km)
        """
        R = self.config['earth_radius_km']
        cosine_val = (np.sin(lat1) * np.sin(lat2) +
                      np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))
        cosine_val = np.clip(cosine_val, -1.0, 1.0)
        central_angle = np.arccos(cosine_val)
        return R * central_angle

    def calculate_tilt(self, cross_track_km, altitude_km):
        """
        根据地面横向距离 cross_track_km 与卫星高度 altitude_km
        计算侧摆角（正值，单位：度），采用 arctan2 形式
        """
        R = self.config['earth_radius_km']
        delta_sigma = cross_track_km / R  # 星下点与目标的中心角(弧度)
        delta_sigma = max(min(delta_sigma, math.pi), -math.pi)  # 限制在 [-π, π]
        numerator = R * math.sin(delta_sigma)
        denominator = (R + altitude_km) - R * math.cos(delta_sigma)
        tilt_rad = math.atan2(numerator, denominator)
        return math.degrees(tilt_rad)

    def bearing_between_points(self, lat1, lon1, lat2, lon2):
        """
        计算从点1指向点2的方位角，输入均为弧度，输出为0~360度
        """
        dLon = lon2 - lon1
        y = math.sin(dLon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
        brng = math.degrees(math.atan2(y, x))
        return (brng + 360) % 360

    def calculate_tilt_signed(self, satellite, observation_time, target_lat_deg, target_lon_deg):
        """
        计算带符号的侧摆角。正值表示目标在卫星运动方向右侧，负值表示左侧。
        """
        lat_sat_rad, lon_sat_rad, alt_sat_km, velocity_east, velocity_north = satellite.get_state_with_velocity(
            observation_time)
        track_bearing_deg = math.degrees(math.atan2(velocity_east, velocity_north)) % 360

        target_lat_rad = math.radians(target_lat_deg)
        target_lon_rad = math.radians(target_lon_deg)
        bearing_to_target = self.bearing_between_points(lat_sat_rad, lon_sat_rad, target_lat_rad, target_lon_rad)
        angle_diff = (bearing_to_target - track_bearing_deg + 180) % 360 - 180

        cross_track_km = self.great_circle_distance_km(lat_sat_rad, lon_sat_rad, target_lat_rad, target_lon_rad)
        tilt_mag = self.calculate_tilt(cross_track_km, alt_sat_km)
        tilt_signed = tilt_mag * math.copysign(1, angle_diff)
        return tilt_signed

    def find_transit_events(self, satellite, target_point, start_time, end_time):
        """
        检测卫星在目标点附近的过境事件（升起、顶点、落下）
        返回过境窗口列表：[(rise_time, set_time), ...]
        """
        events = satellite.satellite.find_events(
            target_point,
            self.ts.from_datetime(start_time),
            self.ts.from_datetime(end_time),
            altitude_degrees = self.config['min_elevation_deg']
        )
        windows = []
        times, event_types = events
        for i in range(0, len(times) - 2, 3):
            rise = times[i].utc_datetime()
            set = times[i + 2].utc_datetime()
            windows.append((rise, set))
        return windows

    def find_all_closest_approaches(self, satellite, target_pos, start_time, end_time):
        """
        分过境窗口搜索垂点，返回所有满足侧摆角要求的时刻
        """
        target_point = wgs84.latlon(target_pos[0], target_pos[1])
        transit_windows = self.find_transit_events(satellite, target_point, start_time, end_time)
        all_approaches = []

        for window_start, window_end in transit_windows:
            # 在单个过境窗口内搜索垂点
            best_time, tilt = self.find_closest_approach_time(
                satellite, target_pos, window_start, window_end
            )
            if best_time and abs(tilt) <= self.config['max_side_look_deg']:
                all_approaches.append((best_time, tilt))

        return all_approaches

    def find_closest_approach_time(self, satellite, target_pos, start_time, end_time, N = 200):
        target_lat_deg, target_lon_deg = target_pos
        target_lat_rad = np.radians(target_lat_deg)
        target_lon_rad = np.radians(target_lon_deg)

        total_seconds = (end_time - start_time).total_seconds()
        # 粗扫：增加采样点至200
        offsets = np.linspace(0, total_seconds, N)
        coarse_times = [start_time + timedelta(seconds = float(s)) for s in offsets]
        t_vec = self.ts.from_datetimes(coarse_times)

        # 向量化计算星下点
        geocentric = satellite.satellite.at(t_vec)
        subpoints = wgs84.subpoint(geocentric)
        lat_array = subpoints.latitude.radians
        lon_array = subpoints.longitude.radians

        # 计算所有时刻的距离
        distances = self.great_circle_distance_km(lat_array, lon_array, target_lat_rad, target_lon_rad)
        min_index = np.argmin(distances)
        coarse_best_time = coarse_times[min_index]

        # 局部三分搜索：增加动态收敛判断
        delta = offsets[1] if N > 1 else 0
        t_low = max(coarse_best_time - timedelta(seconds = delta), start_time)
        t_high = min(coarse_best_time + timedelta(seconds = delta), end_time)

        prev_min_distance = float('inf')
        for _ in range(20):
            diff = (t_high - t_low).total_seconds()
            if diff < 0.1:
                break
            m1 = t_low + timedelta(seconds = diff / 3)
            m2 = t_low + timedelta(seconds = 2 * diff / 3)
            d_m1 = self.great_circle_distance_km(*satellite.get_state(m1)[:2], target_lat_rad, target_lon_rad)
            d_m2 = self.great_circle_distance_km(*satellite.get_state(m2)[:2], target_lat_rad, target_lon_rad)

            # 动态收敛判断
            current_min = min(d_m1, d_m2)
            if abs(prev_min_distance - current_min) < 1e-6:  # 变化小于1米
                break
            prev_min_distance = current_min

            if d_m1 < d_m2:
                t_high = m2
            else:
                t_low = m1

        best_time_refined = t_low + (t_high - t_low) / 2
        tilt_signed = self.calculate_tilt_signed(satellite, best_time_refined, target_lat_deg, target_lon_deg)
        return best_time_refined, tilt_signed

    def analyze_satellite(self, satellite, target_point, start_time, end_time):
        """
        分析单颗卫星的所有过境窗口，返回符合条件的机会列表
        """
        approaches = self.find_all_closest_approaches(
            satellite,
            (target_point.latitude.degrees, target_point.longitude.degrees),
            start_time,
            end_time
        )
        opportunities = []
        for best_time, tilt in approaches:
            opportunities.append((best_time, satellite.name, tilt))
        return opportunities if opportunities else None

    def find_continuous_observations(self, satellites, target, start_time, end_time):
        """
        收集所有卫星在给定时段内的可视机会，返回按时刻排序的列表。
        """
        target_point = wgs84.latlon(target[0], target[1])
        all_ops = []
        for sat in satellites:
            op = self.analyze_satellite(sat, target_point, start_time, end_time)
            if op:
                all_ops.append(op)
        all_ops.sort(key = lambda x: x[0])
        return all_ops

    def process_tasks(self, tasks_json, satellites):
        """
        任务处理主入口
        :param tasks_json: 任务字典
        :param satellites: Satellite对象列表
        :param camera_params: 相机参数
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
                (target_lat, target_lon),
                start,
                end
            )
            passes = []
            if ops:
                for op in ops:
                    if op:  # 确保op不是空列表
                        pass_time, sat_name, tilt = op[0]  # 取出第一个元素
                        passes.append({
                            "satellite": sat_name,
                            "pass_time": pass_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "side_look_angle": round(tilt, 2)
                        })

                # 无论是否有结果都返回passes字段，保持结构一致
            results[task_id] = {
                "passes": passes if passes else []  # 确保总是返回列表
            }

            return results


if __name__ == "__main__":

    tasks_json = {
        "1": {
            "location": "杭州",
            "LLA": [120.1533, 30.2484, 0.007],
            "Validity_period": ["2025-03-24 08:0:0.00000", "2025-03-24 18:0:0.00000"],
            "Observation_mode": "continuous"
        }
    }

    tle_json = {"SAT-01":
                    ('1 99999U 24330.93462963  .00000215  00000-0  10961-4 0 00001',
                     '2 99999 097.3992 296.7393 0013211 261.9647 283.6961 15.16828644000019'),
                "SAT-02":
                    ('1 99999U 24330.93462963  .00000252  00000-0  12839-4 0 00008',
                     '2 99999 097.3992 296.7393 0013200 261.9045 238.7564 15.16828577000014'),
                "SAT-03":
                    ('1 99999U 24330.93462963  .00000280  00000-0  14259-4 0 00007',
                     '2 99999 097.3992 296.7393 0013195 261.8916 223.7693 15.16828532000011'),
                "SAT-04":
                    ('1 99999U 24330.93462963 .00000306 00000-0 15580-4 0 00004',
                     '2 99999 097.3992 296.7393 0013190 261.8822 208.7788 15.16828496000019'),
                "SAT-05":
                    ('1 99999U 24330.93462963  .00000323  00000-0  16443-4 0 00002',
                     '2 99999 097.3992 296.7393 0013183 261.8781 193.7828 15.16828482000018'),
                "SAT-06":
                    ('1 99999U 24330.93462963  .00000326  00000-0  16572-4 0 00008',
                     '2 99999 097.3992 296.7393 0013176 261.8825 178.7783 15.16828499000010'),
                "SAT-07":
                    ('1 99999U 24330.93462963  .00000257  00000-0  13088-4 0 00000',
                     '2 99999 097.3992 296.7393 0013204 262.0727 343.5883 15.16828558000015'),
                "SAT-08":
                    ('1 99999U 24330.93462963  .00000251  00000-0  12783-4 0 00005',
                     '2 99999 097.3992 296.7393 0013208 262.0504 328.6106 15.16828571000019'),
                "SAT-09":
                    ('1 99999U 24330.93462963  .00000238  00000-0  12125-4 0 00000',
                     '2 99999 097.3992 296.7393 0013211 262.0228 313.6381 15.16828598000014'),
                "SAT-10":
                    ('1 99999U 24330.93462963  .00000225  00000-0  11430-4 0 00004',
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
    planner = MissionPlanner()
    # 执行任务规划
    results = planner.process_tasks(
        tasks_json = tasks_json,
        satellites = satellites
    )

    # 转换为JSON友好格式
    print(json.dumps(results, indent = 2))
