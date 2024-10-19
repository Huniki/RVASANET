import cv2
import os

def pic_to_vid(P, V, F):
    path = P
    video_dir = V
    fps = F
    in_img = os.listdir(path)
    # get_key是sotred函数用来比较的元素，该处用lambda表达式替代函数。
    img_key = lambda i: int(i.split('.')[0])
    img_sorted = sorted(in_img, key=img_key)
    # 需要转为视频的图片的尺寸，这里必须和图片尺寸一致
    # w,h of image
    img = cv2.imread(os.path.join(path, img_sorted[0]))
    img_size = (img.shape[1], img.shape[0])
    # 获取名称
    seq_name = os.path.dirname(path).split('/')[-1]
    video_dir = os.path.join(video_dir, seq_name + '.avi')
    #print(img_size)
    video = cv2.VideoWriter(video_dir, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, img_size)

    for item in img_sorted:
        img = os.path.join(path, item)
        img = cv2.imread(img)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()
    print('全部图片已全部转化为视频。')

#主函数
if __name__ == '__main__':
    # 输入图片路径
    path = '/home/rpf/Pillarseg/OpenPCDet/output/pillarseg/pred_model_seg/vis'
    # 输出视频路径
    video_dir = '/home/rpf/Pillarseg/OpenPCDet/output/pillarseg/pred_model_seg'
    # 跟自己的需求设置帧率
    fps = 5
    # 传入函数，转化视频
    pic_to_vid(path, video_dir, fps)
