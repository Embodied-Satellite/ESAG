import os
import json
from typing import List

from agno.agent import Agent
from agno.tools import Toolkit
from agno.utils.log import logger


JSON_FILE_PATH = '/home/mars/cyh_ws/SPA/satellite_data.json'
class SatelliteAbilitiesTool(Toolkit):
    def __init__(self):
        super().__init__(name="Satellite Abilities Tool")
        self.register(self.run_tool)

    def run_read_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False)

    
    def run_tool(self, query: str):
        """
        The tool reads a JSON file to get the satellite capabilities and then selects the satellite that is best suited for the mission.
        
        Args:
            query (str): The query to be processed. It should contain the satellite capabilities and the mission details.
            
        Returns:
            satellite_id (str): The satellite ID.
            target_area (str): The target area for observation.
            observation_type (str): The type of observation.
            onboard_resources (dict): The parameters of onboard resources.
            plan result (str): JSON content.
        
        """
        
        if not os.path.exists(JSON_FILE_PATH):
            return "Capabilities file not found."
        
        json_str = self.run_read_json(JSON_FILE_PATH)
        
        
        return json_str
