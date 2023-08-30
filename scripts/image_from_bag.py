import rosbag
import cv2
from cv_bridge import CvBridge

bridge = CvBridge()

# data_idx = 1
for data_idx in range(7):
    bag_dir = "/home/kai/rosbags/nv_ext_calib/bags/"
    bag_name = "{}.bag".format(data_idx)
    image_topic = "/mvsua_cam/image_raw1"

    bag =rosbag.Bag(bag_dir + bag_name)

    idx = 0

    for topic, msg, t in bag.read_messages(topics=image_topic):
        idx += 1
        if idx==60:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            cv2.imwrite("{}.bmp".format(data_idx), cv_img)
            break
