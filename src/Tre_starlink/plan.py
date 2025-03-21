import numpy as np
from deap import algorithms, base, creator, tools
import multiprocessing
import json
import time
from src.Tre_starlink.util import convert_to_native_types,send_command_via_tcp
from datetime import datetime

np.random.seed(42)

class GeneticAlgorithm:
    def __init__(self, starttime,Planning_duration = 86400):
        self.starttime = starttime
        self.Planning_duration = Planning_duration

    def load_tasks_from_file(self,path):
        # 重新加载tasks_info和state_info
        with open(path, 'r') as f:
            data = json.load(f)
            tasks_input = data['tasks']
            tasks_info = data['tasks_info']
            state_info = data['state_info']

        state_info = {int(key): value for key, value in state_info.items()}
        # 初始化任务数据（示例）
        tasks = []
        for task in tasks_info:
            task_type = int(tasks_input[int(task[0])][9])
            gains = task[1]
            task_id = int(task[0])
            tasks.append({
                'type': task_type,
                'gains': gains,
                'task_id': task_id,  # 任务全局id
                'task_priority': tasks_input[task_id][6],
                'time_priority': tasks_input[task_id][7],
                'Quality_priority': tasks_input[task_id][8],
            })
        self.tasks = tasks
        self.init_state = state_info
        self.init_state.keys()
        self.task_nums = len(tasks)
        return tasks
    
    def load_tasks(self,tasks_input, tasks_info, state_info):
        # 初始化任务数据（示例）
        tasks = []
        for task in tasks_info:
            task_id = task[0]
            index = int(np.where(tasks_input[:,0] == task_id)[0])
            task_type = int(tasks_input[index][9])
            gains = task[1]
            task_id = int(task[0])
            tasks.append({
                'type': task_type,
                'gains': gains.tolist(),
                'task_id': task_id,  # 任务全局id
                'task_priority': tasks_input[index][6],
                'time_priority': tasks_input[index][7],
                'Quality_priority': tasks_input[index][8],
            })
        self.tasks = tasks
        self.init_state = state_info
        self.task_nums = len(tasks)
        return tasks
    # 产生随机基因组
    # 基因编码规则：
    # 类型1任务：0-4表示选择的子任务
    # 类型2任务：5位二进制码表示子任务选择（如10101表示选择0,2,4号任务）
    def create_gene(self,task):
        nums = len(task['gains'])
        if task['type'] == 1:
            return [np.random.randint(0, nums),nums]
        else:
            return [[np.random.randint(0, 2) for _ in range(nums)],nums]

    #变异操作定义
    def mutation(self,ind):
        # 随机选择变异点
        mut_idx = np.random.randint(len(ind))
        task = self.tasks[mut_idx]
        
        if task['type'] == 1:
            # 类型1任务：随机选择新子任务
            ind[mut_idx][0] = np.random.randint(0, ind[mut_idx][1])
        else:
            # 类型2任务：翻转一个随机位
            bit = np.random.randint(ind[mut_idx][1])
            ind[mut_idx][0][bit] = 1 - ind[mut_idx][0][bit]
        return ind,

    # 遗传操作定义
    def crossover(self,ind1, ind2):
        # 两点交叉
        size = min(len(ind1), len(ind2))
        cx1, cx2 = sorted(np.random.choice(range(size), 2, replace=True))
        
        for i in range(cx1, cx2):
            if isinstance(ind1[i], list):  # 类型2任务需要特殊处理
                ind1[i], ind2[i] = ind2[i].copy(), ind1[i].copy()
            else:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2
    
    #ind基因组转卫星任务
    def get_satellite_task(self,individual):
        satellite_tasks=[[] for i in range(12)]
        for i, gene in enumerate(individual):
            gene = gene[0]
            task = self.tasks[i]
            # 单次执行的点
            if task['type'] == 1:
                satellite_id = int(task['gains'][gene][0])
                satellite_tasks[satellite_id].append([task['task_id']]+task['gains'][gene][1:3])
            else:
                for j in range(len(gene)):
                    if gene[j]:
                        satellite_id = int(task['gains'][j][0])
                        satellite_tasks[satellite_id].append([task['task_id']]+task['gains'][j][1:3])

        for i, satellite_task in enumerate(satellite_tasks):
            satellite_task = np.array(satellite_task)
            #卫星任务按时间排序
            # print(satellite_task.shape)
            if satellite_task.size != 0:
                sort_indices = np.argsort(satellite_task[:,1])
                satellite_tasks[i] = satellite_task[sort_indices]
            else:
                satellite_tasks[i] = satellite_task
        return satellite_tasks

    def task_gain(self,task,starttime,j):
        # 考虑任务,时间,质量权重
        image_qulity = (task['gains'][j][-1])*task['Quality_priority']/10
        time_gain = (1-(task['gains'][j][1] - starttime)/self.Planning_duration)*task['time_priority']/10
        gain = (image_qulity + time_gain)*task['task_priority']
        return gain
    
    # 适应度计算
    def calculate_fitness(self,individual,logger = False):
        total = 0
        
        # 第一步：计算各任务自身增益:拍摄完成时间\质量
        for i, gene in enumerate(individual):
            gene = gene[0]
            task = self.tasks[i]
            # 单次执行的点
            if task['type'] == 1:
                total += self.task_gain(task,self.starttime,gene)
            else:
                for j in range(len(gene)):
                    if gene[j]:
                        total += self.task_gain(task,self.starttime,j)

        satellite_tasks = self.get_satellite_task(individual)

        # 第二步：计算任务间协同增益（示例公式）
        #考虑负载均衡和拍摄能耗效率
        satellites_energy = []
        energy_scale = 0.5
        for i, satellite_task in enumerate(satellite_tasks):
            if satellite_task.size !=0:
                #接入卫星初始状态
                # print(type(i))
                init_state = self.init_state[i+1]
                satellite_task = np.vstack((np.array([[init_state['current_time'],init_state['roll_angle']/180*3.14]]),satellite_task[:,1:]))
                #计算任务的侧摆角和侧摆速度
                roll_angle = np.abs(satellite_task[:,1][1:]-satellite_task[:,1][:-1])
                roll_time = satellite_task[:,0][1:]-satellite_task[:,0][:-1]
                roll_time_need = roll_angle/3.1415*180/0.5 

                #计算卫星侧摆耗能
                roll_energy = np.sum(roll_angle)/3.14*180*0.01*energy_scale  #转1度耗能0.01
                #根据拍照次数统计拍照能量
                photo_energy = len(satellite_task)*0.012*energy_scale   #拍一张照片能耗0.012 
                satellite_energy = roll_energy + photo_energy

                #侧摆速度超过卫星最大侧摆速度,规划失败
                if max(roll_time_need - roll_time + 5) > 0:   #5秒的稳定时间(侧摆5秒后才能拍照)
                    total = total/100
                    if logger:
                        print(f'satellite{i}侧摆速度超过限制')
                    # break

                #24小时总能耗超过能耗限制
                if satellite_energy > 300:
                    total = total/100
                    if logger:
                        print(f'satellite{i}总能耗超过能耗限制')
                    # break

                satellites_energy.append(satellite_energy)

        #计算能耗均衡增益
        # if total != 0:
        total = total - sum(satellites_energy)    #能量损耗
        energy_std = np.std(satellites_energy)*energy_scale
        total = total - energy_std             #能量均衡损耗

        return total,
    
    def plan(self):
        # 定义遗传算法框架
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        # 初始化种群
        toolbox.register("individual", tools.initIterate, creator.Individual,lambda: [self.create_gene(task) for task in self.tasks])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.calculate_fitness)
        toolbox.register("mate", self.crossover)
        toolbox.register("mutate", self.mutation)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # 设置并行池
        pool = multiprocessing.Pool(processes=24)
        # 注册并行映射函数
        toolbox.register("map", pool.map)

        # 运行优化
        POPULATION_SIZE = 1000 # 种群大小 len(self.tasks)*5
        MAX_GENERATION = self.task_nums+10   # 迭代次数
        CXPB = 0.7           # 交叉概率
        MUTPB = 0.3          # 变异概率
        population = toolbox.population(n=POPULATION_SIZE)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)

        result, _ = algorithms.eaSimple(
            population,
            toolbox,
            cxpb=CXPB,
            mutpb=MUTPB,
            ngen=MAX_GENERATION,
            stats=stats,
            verbose=True
        )

        # 关闭并行池
        pool.close()
        pool.join()

        # 输出结果
        best_ind = tools.selBest(result, k=1)[0]
        print(f"最优总增益: {best_ind.fitness.values[0]}")

        #结果验证
        self.calculate_fitness(best_ind,logger=True)

        # 输出
        self.satellite_tasks = self.get_satellite_task(best_ind)
        with open('/home/mars/cyh_ws/ESAG/src/Tre_starlink/dataset/satellite_task_plan.json', 'w') as f:
            data = {'satellite_task':convert_to_native_types(self.satellite_tasks)}
            json.dump(data, f)


    def execute(self,):
        # with open('/home/yuyuan/Project/knowledge_graph/Tre_starlink/dataset/satellite_task_plan.json', 'r') as f:
        #     data = json.load(f)
        #     satellite_tasks = data['satellite_task']

        # with open('/home/yuyuan/Project/knowledge_graph/Tre_starlink/dataset/tasks_state_info.json', 'r') as f:
        #     data = json.load(f)
        #     tasks_input = data['tasks']
        #     tasks_info = data['tasks_info']
        #     state_info = data['state_info']

        satellite_tasks = self.satellite_tasks
        state_info = self.init_state

        # 计算侧摆动作时间
        satellite_id = 3
        commands_execute = {
            "satellite_cur": str(satellite_id),
            "task": {}
        }

        for i, satellite_task in enumerate(satellite_tasks):
            if len(satellite_task) != 0:
                satellite_task = np.array(satellite_task)
                task_ids = satellite_task[:, 0]
                photo_time = satellite_task[:, 1]
                target_angel = satellite_task[:, 2] / 3.1415 * 180
                init_state = state_info[i + 1]
                satellite_task = np.vstack((np.array([[init_state['current_time'], init_state['roll_angle'] / 180 * 3.14]]), satellite_task[:,1:]))
                roll_angle = satellite_task[1:, 1] - satellite_task[:-1, 1]
                roll_time = np.abs(roll_angle) * 180 / 3.1415 / 0.5
                roll_execute_time = satellite_task[1:, 0] - roll_time - 5  # 5秒准备时间
                commands = np.vstack((task_ids, photo_time, roll_execute_time, target_angel)).T
                command_execute = []
                for command in commands:
                    task_id = int(command[0])
                    time_format = "%Y-%m-%d %H:%M:%S.%f"
                    photo_time = datetime.fromtimestamp(command[1]).strftime(time_format)[:-3]
                    roll_time = datetime.fromtimestamp(command[2]).strftime(time_format)[:-3]
                    command_execute.append("T:%s C:SET_ATTITUDE P:%f" % (roll_time, command[3]))
                    command_execute.append("T:%s C:GET_IMAGE P:%i" % (photo_time,task_id))

                commands_execute["task"][str(i + 1)] = command_execute

        commands_execute_str = json.dumps(commands_execute)
        # commands_execute_str = commands_execute_str.replace("\"", '\\"')

        json.loads(commands_execute_str)

        commands_execute_str = "TASK_UPDATE "+commands_execute_str
        # 新增功能：通过 TCP/IP 发送命令
        send_command_via_tcp(commands_execute_str,port=5000+satellite_id)
        return commands_execute_str

if __name__ == "__main__":
    timestamp = '2025-03-07 5:0:14.131254'
    starttime = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    path = '/home/mars/cyh_ws/ESAG/src/Tre_starlink/dataset/tasks_state_info_50.json'
    time_start = time.time()
    GA = GeneticAlgorithm(starttime=starttime)
    GA.load_tasks_from_file(path)
    GA.plan()
    GA.execute()
    time_end = time.time()
    print("规划耗时:",time_end-time_start)


# 种群数=500,迭代次数 = 300,最优总增益=13.055,time=88.515
# 种群数=1000,迭代次数 = 180,最优总增益=42.574,time=121
# 种群数=2000,迭代次数 = 160,最优总增益=40.1478,time=221
