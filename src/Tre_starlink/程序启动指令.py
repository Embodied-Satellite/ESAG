#仿真
setconda
conda activate sg 
python '/home/yuyuan/Project/knowledge_graph/Tre_starlink/simulate.py'

#显示
SHOW=true /home/yuyuan/blender/blender-3.5.1-linux-x64/blender  --python '/home/yuyuan/Project/knowledge_graph/Tre_starlink/blender_sim.py' 
#后台仿真
SHOW=false /home/yuyuan/blender/blender-3.5.1-linux-x64/blender -b --python '/home/yuyuan/Project/knowledge_graph/Tre_starlink/blender_sim.py' 


#状态监控
watch -n 0.5 "
echo "GET_STATE" | nc 127.0.0.1 5001;
echo;
echo "GET_STATE" | nc 127.0.0.1 5002
echo;
echo "GET_STATE" | nc 127.0.0.1 5003
echo;
echo "GET_STATE" | nc 127.0.0.1 5004
echo;
echo "GET_STATE" | nc 127.0.0.1 5005
echo;
echo "GET_STATE" | nc 127.0.0.1 5006
echo;
echo "GET_STATE" | nc 127.0.0.1 5007
echo;
echo "GET_STATE" | nc 127.0.0.1 5008
echo;
echo "GET_STATE" | nc 127.0.0.1 5009
echo;
echo "GET_STATE" | nc 127.0.0.1 5010
echo;
echo "GET_STATE" | nc 127.0.0.1 5011
echo;
echo "GET_STATE" | nc 127.0.0.1 5012
"
