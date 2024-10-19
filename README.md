# RVSAD-Net
### 4D Radar And Vision Fusion Detection Model Based On Segmentation-assisted

## The Illustration of RVSAD-Net
### RVSAD-Net Model Framework.
![fig5-1](https://github.com/user-attachments/assets/2ef1c0c9-58c9-4f97-b878-e7cf68690dd9)

### Flow diagram of the radar reference point module.
![fig6](https://github.com/user-attachments/assets/96cc3d12-d852-4b98-bc19-2a4000e70339)

## Contributions
• A 4D radar detection model is proposed based on BEV point cloud segmentationassisted target detection. Improve the performance of point cloud detection by segmenting auxiliary supervision.
• The 4D radar data in the VoD dataset are comprehensively counted, and the radar reference point method is designed to deal with the problem of sparse point clouds and no radar points in some 3D label boxes.
• An attention-based multi-dimensional feature fusion module is proposed, which makes full use of the velocity, RCS information and semantic category information of 4D radar point cloud data.


## Details
### Point clouds projection map visualization to qualitative analysis. The original point clouds are filtered by 3D label boxes to obtain segmentation labels. (a) is lidar point cloud projection. (b) is 4D radar point cloud projection.
![9912cbaad7da42f769d4cb01033dec6](https://github.com/user-attachments/assets/0a4b9f01-3e7c-48f7-8ed2-9433a33ef3fb)

### Bird’s Eye View segmentation label visualization.
![0e9a058541d40de0f264aeda7706109](https://github.com/user-attachments/assets/c7c391bc-75c9-4bf1-8802-686ccab71eb9)

### Bird’s Eye View segmentation label visualization.
![6183b88407697c4e092dd304dd97f97](https://github.com/user-attachments/assets/1f73a778-043b-4401-be33-0ad0365fbf5b)

## Results
### Model comparison sample visualization.
![8e5350a7048dff802e36341fcb947e5](https://github.com/user-attachments/assets/7f9152c9-dcba-45e0-a2f5-3b40c0b43599)

###
#### Comparison of the improved algorithm with the classical algorithm
##### Radar 5frames_Entire annotated area
|method|Car(%)| Ped(%) |Cyc(%) | mAP(%) | 
| ------------ | --------- | ----- | ------- | ----------- | 
|              | 39.32      |30.24 | 63.81    |44.46        |
| SA           | 40.41      | 32.70  | 67.11    | 46.74       | 
| RR           | 40.56     | 32.52  |66.61    | 46.56        | 
| PA           | 38.95      | 31.330  | 65.54    |45.26        | 
| SA+RR        | 40.73      | 37.58  | 64.59    |47.64       |
| SA+PA        | 37.94      | 37.07  |67.46    | 47.49        | 
|  SA+RR+PA   |40.52      | 38.15  | 66.08    | 48.25        | 

#####  Radar 5frames_In driving corridor
|method|Car(%)| Ped(%) |Cyc(%) | mAP(%) |
| ------------ | --------- | ----- | ------- | ----------- | 
|              | 71.72      |41.06 | 84.86    |65.88        |
| SA           | 71.75      | 43.11  | 87.61    | 67.49       | 
| RR           | 72.36     | 43.49  |87.89    | 67.91        | 
| PA           | 71.29      |42.01  | 87.58    |67.13        | 
| SA+RR        | 72.05      | 48.71  | 88.57    |69.78       |
| SA+PA        | 71.76     | 47.47  |87.38    | 68.87        | 
|  SA+RR+PA   |71.36      |50.09  | 89.69    |70.47        | 

##### Lidar_Entire annotated area
|method     |Car(%)      | Ped(%)     |Cyc(%) | mAP(%) |
| ------------ | --------- | ----- | ------- | ----------- | 
|             | 60.03      |32.19 | 62.93    |51.72        |
|     SA      | 61.29      |47.92 | 66.74    |58.65        |

##### Lidar_In driving corridor
|method|Car(%)| Ped(%) |Cyc(%) | mAP(%) |
| ------------ | --------- | ----- | ------- | ----------- | 
|             | 60.03      |32.19 | 62.93    |51.72        |
|     SA      | 61.29      |47.92 | 66.74    |58.65        |
