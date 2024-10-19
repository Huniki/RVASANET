import cv2
img_path = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/lidar/training/image_2/06669.jpg'
img = cv2.imread(img_path)
print(img.shape)