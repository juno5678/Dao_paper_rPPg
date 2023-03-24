import rosbag

filename = r"D:\CCU\HR_Estimator\other_HR_Estimator\Japan_Patch_rPPG\RealSense Frameset 10.bag"
bag = rosbag.Bag(filename, 'r')

topic = bag.get_type_and_topic_info()[1].keys()
print(topic)

#cnt_depth = info.topics.get('/device_0/sensor_0/Infrared_1/image/data').message_count
#print('infrared image count: ', cnt_depth)
#
#
#cnt_color = info.topics.get('/device_0/sensor_1/Color_0/image/data').message_count
#print('color image count: ', cnt_color)


bag_data = bag.read_messages('/device_0/sensor_0/Infrared_1/image/data')

t_tmp = 0
t = 0

for topic, msg, t in bag_data:
    print(msg.header.stamp.to_sec(), t.to_nsec()-t_tmp)
    t_tmp = t.to_nsec()
