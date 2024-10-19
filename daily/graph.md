# 图
共需要完成以下：
- [ ] 原始radar投影到图像
- [ ] 原始radar的BEV图
- [ ] 原始radar的BEV标签图
- [ ] 原始lidar的BEV标签图
- [ ] 原始lidar投影到图像
- [ ] 原始radar的数据统计图
- [ ] 原始lidar的数据统计图
- [ ] 算法流程图，整体框架
- [ ] 分割模块图
- [ ] 雷达参考点模块图
- [ ] 注意力模块图
- [ ] 可视化结果
分步完成：
# 2024.3.1
计划：
- [ ] 原始radar投影到图像
- [ ] 原始radar的BEV图
- [ ] 原始radar的BEV标签图
- [ ] 原始lidar的BEV标签图
- [ ] 原始lidar投影到图像
前提：
- [ ] dataset同时读取lidar、radar、图像等数据
- [ ] 投影部分代码书写
- [ ] 确定lidar和radar是否在同一坐标系
旋转平移矩阵：整个数据集只有一个旋转平移矩阵，Tr_velo_to_cam！
**【注】** 作者没有统一radar和lidar的坐标系，因此lidar2 cam和radar2cam是不一样的。
问题：在3维目标检测中，用的谁的坐标系。
不同的数据生成了不同的pkl！所以在生成radar的pkl的时候，只保留了radar的旋转平移矩阵，因此如果需要lidar，就需要生成lidar的pkl