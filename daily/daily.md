# 2024.1.14
2024.1.14今天开始对模型进行训练，首先训练的就是用pointpillar对vod数据集进行训练。这一步之前是做过的。我们继续之前的事情。

后续需要做的事情
1、对训练结果进行测试
2、对齐vod论文中的评价指标和目前现在使用的评价指标
3、将检测头换成anchor free
4、为pillar分割做准备：a，需要得到BEV下的分割label，然后参考heatmap的结构，将二者做一个联系。
# 2024.1.15
TODO：
1、outputs保存地址改一下。
2、驾驶区域的评价指标怎么做？
3、类别合并的事情
4、POINT_CLOUD_RANGE: [0, -25.6, -3, 51.2, 25.6, 1]、体素大小也需要更改、不要过滤没有点或者点少的样本、使用所有雷达功能，使用雷达时，不需要使用旋转的数据增强
5、 turn off the use of ground planes
6、80epoch
7、官方论文中5frame radar结果car 44.8、ped 42.1、cyclist 54.0 
目前测试3dbox：
car 46.365、ped 35.98、cyclist 66.8349 
8、pkl中存在四個，train包含了5139個數據，test包含了2247個數據，val包含了1296個數據，trainval包含了6435。test數據集中沒有標籤，不能使用。
# 2024.1.16
1、确定大论文章节名称
第一章 绪论
第二章 多传感器融合算法框架与实车验证平台搭建
第三章 基于图像输入的多任务路面信息提取模型
第四章 基于多模态的BEV障碍物检测模型
第五章 基于多任务的融合算法
第六章 模型部署及实车测试
# 2024.1.17
1、点云BEV可视化+box可视化
# 2024.1.24
- [ ] 1、更换检测头尝试：anchor base--> anchor free。 
- [ ] 2、点云分割的代码尝试写一下。标签部分需要写一下。
- [ ] 3、outputs保存地址改一下。
# 2024.1.25
- [x] 1、更换检测头尝试：anchor base--> anchor free。
这部分更换后难以收敛，效果比较差
- [x] 2、点云分割的代码尝试写一下。标签部分需要写一下。
这部分需要做的：那部分划分的pillar的代码是self.prepare_data这里面做的pillar划分。pillar划分的函数在data_processor.py中的transform_points_to_voxels函数中。如何在这个函数中得到seglabel。
提示：在生成voxel_label时，使用的coordinates对每个voxel_points进行定义，voxel_points是根据coordinates中的voxel坐标确定xy位置的点，然后判断哪些点在gt_box中，然后给定类别。最终输出voxel_label。这里需要注意background背景的问题。
目前的grid size是[320,320]
voxels的含义：[voxel个数，每个voxel中的点的个数，每个点的特征]
coordinates表示每个voxel的坐标。
正样本框和负样本框不平衡的问题也能得到解决。
- [x] 3、outputs保存地址改一下。
- [ ] 4、分割辅助检测的思路需要理一下。
- [ ] 重新搭建一个模型，这个模型做分割。主要包括下面一些需要待解决的问题
- [ ] 模型的搭建。
模型部分，pointpillar包括了四个部分，vfe、pillarscatter、backbone2d、anchorhead。
vfe--->获得pillar_feature--->[pillar个数，64]
pillar scatter---->获得spatial features --->[bs，64，320，320]
backbone2d --->卷积下采样，这里仅仅做了二倍下采样。[bs，384，160，160]
head--->
这里往后就应该接Unet模型，进行分割！
- [x] loss的设置
这里的loss返回了三个部分，在pointpillar中，loss是总计损失。其余两个部分是没有很大用处的，这里loss配置了ce_loss作为监督。目前最大bs可以为6。到这里整个train部分基本没问题啦。
- [ ] 评价指标的设置
- [ ] 测试结果可视化等
- [ ] 5、点云分割的模型需要跑一下。
- [x] instance label去掉这里不考虑实例分割！
- [ ] 6、centerfusion代码是做毫米波的吗？是的话看一下
# 2024.1.26
- [x] pillarseg分割模型搭建完成，可以进行训练。显存占用很大，目前分割模型，loss的降低很快。
- [ ] 评价指标的设置
- [ ] 测试结果可视化等
- [ ] loss函数中增加dice loss进行训练。diceloss计算之前需要对预测结果进行softmax计算。
# 2024.1.31
- [ ] 评价指标的设置
- [ ] 测试结果可视化等
- [ ] loss函数中增加dice loss进行训练。diceloss计算之前需要对预测结果进行softmax计算。
- [ ] Surround Occ模型复现不好复现，代码调试
- [ ] Occ课程第五章
- [ ] 深度学习高手笔记书看一下
- [ ] Pillarseg的小论文框架搭建
- [ ] 小论文实验规划
- [ ] 大论文的框架搭建
- [ ] 上林赋
- [ ] 年度总结

# 2024.2.1
- [x] SurroundOcc代码看一下，如果可以运行的话，跑一下
- [x] Pillarseg分割结果，可视化一下。
- [ ] 大论文使用latex的方法整理下，最好是在windows
- [ ] Pillarseg的相关评价指标，可以做一下。

# 2024.2.2
- [ ] 大论文使用latex的方法整理下，最好是在windows
- [ ] Pillarseg的相关评价指标，可以做一下。
- [ ] 测试数据集看一下，部分异常，考虑要不要把图片或者一些其他的加载可视化中
- [ ] 小论文的思路要开始写了
- [ ] 适当减小下Unet的模型然后看看效果，在加入评价指标之后

# 2024.2.4
- [x] 大论文使用latex的方法整理下，最好是在windows
- [ ] Pillarseg的相关评价指标，可以做一下。
- [x] 测试数据集看一下，部分异常，考虑要不要把图片或者一些其他的加载可视化中
- [x] 将Unet部分添加到配置文件中
- [ ] 配置文件中加入backbone2d及head部分进行目标检测任务
- [ ] 类别合考虑做不做
- [ ] 年前任务就这些---可以包括一些其他实验部分。
# 2024.2.5
- [x] Pillarseg的相关评价指标，可以做一下。
- [x] 配置文件中加入backbone2d及head部分进行目标检测任务
- [ ] 类别合考虑做不做
- [ ] 年前任务就这些---可以包括一些其他实验部分
- [x] dice loss加入
- [ ] 融合模块使用add
- [x] 融合模块使用concat，后通过1卷积改变通道数字（或者通过一定注意力机制操作？）
# 2024.2.9
- [ ] 训练时，做反向传播的logit需不需要做logit或者softmax！