import img_corners_est
import LM_opt
import mid360_corners_est
# print("detect image corners")
# img_corners_est.detect_img_corners()
# print("\ndetect pc corners\n")
# mid360_corners_est.read_rosbag_point_cloud("/home/kai/rosbags/nv_ext_calib/bags/0.bag", "/livox/lidar")
print("\ncalibrate ext params\n")
LM_opt.cal_ext_paras()