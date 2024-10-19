# get_official_eval_result
该函数用来进行kitti官方的评价。这部分需要更改，首先我们使用的是仅三个类别，汽车、行人、自行车。但是因为是使用的毫米波雷达，同时需要按照论文的阈值进行设定。即：汽车：0.5、行人，自行车均为0.25。这里需要更改，所以这里对这个函数进行了简单的学习。
这个函数写在了kitti_dataset.py文件中的class KittiDataset(DatasetTemplate)类中的evaluation函数中。
在get_official_eval_result函数中首先阐述设定了一堆overlap的值。然后根据我们设定的当前包括的类的个数，对应到index。初始定义的类别一共有6个，因为我们只需要三类，也就是[0,1,2]，初始的overlap的shape为(2,3,6)，过滤后就是(2,3,3)，