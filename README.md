# Task Planning Agent for Satellite Constellation

## 项目简介
基于多智能体的卫星任务调度系统，负责从任务生成到任务规划再到任务执行的完整流程。系统通过多个代理协作完成卫星任务的生成、规划和执行，支持星座调度任务。

---

## 功能模块

### 1. **任务生成智能体 (`task_generation_agent`)**
- **功能**：根据用户输入的自然语言指令生成任务请求列表。
- **输入**：
  - 用户指令（如：监测某地交通情况）。
- **输出**：
  - 任务请求列表，包括任务编号、目标位置、任务类型、优先级、48h的云量等。
- **示例输出**：
  
```json
  {
      "task_id": "001",
      "location": "杭州西湖",
      "latitude": 30.25,
      "longitude": 120.155,
      "task_type": "point_target",
      "task_priority": 5,
      "time_priority": 5,
      "quality_priority": 5,
      "validity_period": ["2025-03-26T00:00:00", "2025-03-28T00:00:00"],
      "area_size": null,
      "cloudrate": [  
        1.0, 
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.71,
        0.0,
        0.0,
        0.03,
        0.02,
        0.14,
        0.23,
        0.17,
        0.0,
        0.0,
        0.0,
        0.06,
        0.6,
        0.9,
        0.96,
        0.92,
        0.99,
        1.0,
        0.91,
        0.2,
        0.39,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
  }
```

### 2. **任务规划智能体 (`task_planning_agent`)**
- **功能**：根据任务请求列表生成任务计划。
- **输入**：
  - 任务请求列表。
- **输出**：
  - 任务计划，包括任务编号、任务开始时间、任务结束时间、任务状态等。
- **示例输出**：
  
```json
{
    "task_id": "001",
    "satellite_id": "SAT-01",
    "position": [30.25, 120.155],
    "observation_time": ["2025-03-26T10:00:00", "2025-03-26T10:15:00"],
    "slew_angle": 15,
    "solar_panel_angle": 45
}

```

### 3. **任务执行智能体 (`task_execution_agent`)**
- **功能**：根据任务计划执行卫星任务。
- **输入**：
  - 任务计划。
- **输出**：
  - 任务执行结果，包括任务编号、任务状态、任务执行时间等。
- **示例输出**：
```json
{
    "task_id": "001",
    "satellite_id": "SAT-01",
    "position": [30.25, 120.155],
    "observation_time": ["2025-03-26T10:00:00", "2025-03-26T10:15:00"],
    "slew_angle": 15,
    "solar_panel_angle": 45,
    "status": "completed",
    "execution_time": "2025-03-26T10:05:00"
}
```


##  目录结构
```
ESAG/
├── src/
│   ├── agents/
│   │   ├── task_planning_agent.py # 任务规划多智能体
│   ├── models/
│   │   ├── LLM.py                 # 自定义大语言模型
│   ├── tools/
│   │   ├── satellite_task_plan.py # 卫星任务规划工具
│   │   ├── satellite_task_gen.py  # 卫星任务生成工具
│   │   ├── satellite_task_exec.py # 卫星任务执行工具
│   ├── utils/
│   │   ├── log.py                 # 日志工具
├── logs/                          # 日志文件夹
├── unittests/                     # 测试文件夹
├── app.py                         # app应用程序
├── requirements.txt               # 依赖库
├── README.md                      # 项目说明文档

```

## 使用说明
### 0. **环境配置**
- Python 3.10+
- OLLAMA API: 
```http://10.15.42.153:11434``` （或替换本地大模型）

### 1. **安装依赖**
```bash
pip install -r requirements.txt
```

### 2. **运行应用程序**


```python
python app.py
```

## 代码示例

### 1. **任务生成智能体**
```python
from src.agents.task_generation_agent import TaskGenerationAgent

agent = TaskGenerationAgent()
task_requests = agent.generate_tasks(user_input="监测杭州西湖交通情况")
print(task_requests)
```

### 2. **任务规划智能体**
```python
from src.agents.task_planning_agent import TaskPlanningAgent

agent = TaskPlanningAgent()
task_plans = agent.plan_tasks(task_requests)
print(task_plans)
```

### 3. **任务执行智能体**
```python
from src.agents.task_execution_agent import TaskExecutionAgent

agent = TaskExecutionAgent()
task_results = agent.execute_tasks(task_plans)
print(task_results)
```


## 日志记录
```
ESAG/logs/application.log
```