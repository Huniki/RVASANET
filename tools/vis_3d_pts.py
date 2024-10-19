import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


# import open3d as o3d

# vis = o3d.visualization.Visualizer()
# vis.create_window(width=1080, height=720)
#
# vis.get_render_option().point_size = 1
# vis.get_render_option().background_color = [0, 0, 0]
#
# bin_path = '/home/lightning/work/bishe/VLP16/cap_data/17.bin'
# bin = np.fromfile(bin_path, dtype=np.float32, count=-1).reshape(-1, 4)
# bin_xyz = bin[:, :3]
# pcd = o3d.open3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(bin_xyz)
#
# vis.add_geometry(pcd)
#
# # 画坐标轴，蓝色Z轴 绿色Y轴 红色X轴
# axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
# vis.add_geometry(axis_pcd)
#
# vis.run()
# vis.destroy_window()



####  彩虹色
if __name__ == '__main__':
    # 加载点云数据
    pts_path = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/lidar/training/velodyne/00257.bin'
    number_of_channel = 4
    pts_data = np.fromfile(pts_path, dtype=np.float32).reshape(-1, number_of_channel)
    # points = np.load('/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar_5frames/training/velodyne/00257.bin')
    points = pts_data[:,:3]  # 里面有多个点云  [5, 8192,3]
    pcd = o3d.geometry.PointCloud()  # 传入3d点云

    pcd.points = o3d.utility.Vector3dVector(points)  # point_xyz 二维 numpy 矩阵,将其转换为 open3d 点云格式

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)  # 创建窗口
    render_option = vis.get_render_option()  # 渲染配置
    render_option.background_color = np.array([0, 0, 0])  # 设置点云渲染参数，背景颜色
    render_option.point_size = 2.0  # 设置渲染点的大小
    vis.add_geometry(pcd)  # 添加点云

    vis.run()

    vis.destroy_window()


