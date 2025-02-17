#!/usr/bin/env python

import cv2
import torch
import numpy as np

import rospy
from darknet_ros_msgs.msg import BoundingBoxes
import message_filters
from sensor_msgs.msg import CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from spencer_tracking_msgs.msg import DetectedPersons, DetectedPerson

from deep_sort import build_tracker
from person_tracking.srv import choose_target, clear_target
from utils.draw import draw_boxes

"""
args: display, img_topic, camera, save_results, video_path frame_interval, service

"""


class VideoTracker(object):
    def __init__(self):
        self.detection_id_increment = rospy.get_param('~detection_id_increment', 1)
        self.last_detection_id = rospy.get_param('~detection_id_offset', 0)

        # initialize publishers
        self.image_pub = rospy.Publisher("~image/compressed", CompressedImage, queue_size=1)
        self.spencer_pub = rospy.Publisher("~detected_persons", DetectedPersons, queue_size=1)

        # initialize services to interact with node
        self.target_clear_srv = rospy.Service("~clear_target", clear_target, self.clear_track)
        self.target_choose_srv = rospy.Service("~choose_target", choose_target, self.select_target)

        self.deepsort = None

        self.stored_msgs = {}
        camera_namespaces = rospy.get_param('~camera_namespaces')
        image_topic = rospy.get_param('~image_topic')
        points_topic = rospy.get_param('~points_topic')
        yolo_topic = rospy.get_param('~yolo_topic')
        for i, ns in enumerate(camera_namespaces):
            image_sub = message_filters.Subscriber(ns + image_topic, CompressedImage)
            points_sub = message_filters.Subscriber(ns + points_topic, PointCloud2)
            bbox_sub = message_filters.Subscriber(ns + yolo_topic, BoundingBoxes)
            sub = message_filters.TimeSynchronizer([image_sub, points_sub, bbox_sub], 10)
            if i == 0:
                sub.registerCallback(self.ros_deepsort_callback)
            else:
                sub.registerCallback(self.store_detection)

        self.idx_frame = 0
        self.idx_tracked = None
        self.bbox_xyxy = []
        self.identities = []

    # callback function to clear track
    def clear_track(self, ros_data):
        if self.idx_tracked is not None and ros_data.clear:
            self.idx_tracked = None
            return True
        else:
            return False

    # callback function to select target
    def select_target(self, ros_data):
        if self.idx_tracked is None:
            for identity in self.identities:
                if identity == ros_data.target:
                    self.idx_tracked = ros_data.target
                    return True
            return False
        else:
            return False

    def to_spencer_msg(self, detections, cls_conf, points, shape):
        persons = DetectedPersons()
        height, width = shape[:2]
        for p, conf in zip(detections, cls_conf):
            tlwh = self.deepsort._xyxy_to_tlwh(p[:4])
            x, y = int(tlwh[0] + tlwh[2] / 2.0), int(tlwh[1] + tlwh[3] / 2.0)
            centers = [[xx, yy] for yy in range(max(y - 3, 0), min(y + 3, height))
                       for xx in range(max(x - 3, 0), min(x + 3, width))]
            pts = [p for p in pc2.read_points(points, ('x', 'y', 'z'), uvs=centers, skip_nans=True)]
            if not pts:
                continue
            pt = np.mean(pts, axis=0)
            person = DetectedPerson()
            person.modality = DetectedPerson.MODALITY_GENERIC_RGBD
            person.pose.pose.position.x = pt[0]
            person.pose.pose.position.y = pt[1]
            person.pose.pose.position.z = pt[2]
            person.confidence = conf
            person.reidentification_id = p[-1]
            person.detection_id = self.last_detection_id
            self.last_detection_id += self.detection_id_increment
            large_var = 999999999
            pose_variance = 0.05
            person.pose.covariance[0 * 6 + 0] = pose_variance
            person.pose.covariance[1 * 6 + 1] = pose_variance
            person.pose.covariance[2 * 6 + 2] = pose_variance
            person.pose.covariance[3 * 6 + 3] = large_var
            person.pose.covariance[4 * 6 + 4] = large_var
            person.pose.covariance[5 * 6 + 5] = large_var
            persons.detections.append(person)

        return persons

    def store_detection(self, color, points, bbox):
        self.stored_msgs[points.header.frame_id] = [color, points, bbox]

    def ros_deepsort_callback(self, color, points, bbox):
        if self.spencer_pub.get_num_connections() == 0 and self.image_pub.get_num_connections() == 0:
            if self.deepsort is not None:
                # Free gpu memory
                del self.deepsort
                self.deepsort = None
            return
        if self.deepsort is None:
            # Only construct instance.
            rospy.loginfo('loading deepsort weight.')
            config = {"DEEPSORT": rospy.get_param(rospy.get_name())}
            self.deepsort = build_tracker(config, use_cuda=True)
            return

        ori_im = cv2.imdecode(np.fromstring(color.data, np.uint8), flags=cv2.IMREAD_COLOR)

        bbox_xywhs = []
        cls_confs = []
        ims = []
        self.store_detection(color, points, bbox)
        for c, _, b in self.stored_msgs.values():

            # convert ros compressed image message to opencv
            np_arr = np.fromstring(c.data, np.uint8)
            im = cv2.cvtColor(cv2.imdecode(np_arr, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh = []
            cls_conf = []
            cls_ids = []
            for bb in b.bounding_boxes:
                bbox_xywh.append([bb.x + bb.w / 2, bb.y + bb.h / 2, bb.w, bb.h])
                cls_conf.append(bb.prob)
                cls_ids.append(0 if 'person' in bb.Class else 1)

            if bbox_xywh:
                bbox_xywh = np.array(bbox_xywh)
                cls_conf = np.array(cls_conf)
                cls_ids = np.array(cls_ids)
            else:
                bbox_xywh = torch.FloatTensor([]).reshape([0, 4]).numpy()
                cls_conf = torch.FloatTensor([]).numpy()
                cls_ids = torch.LongTensor([]).numpy()

            # select person class
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            bbox_xywh[:, 2:] *= 1.2
            cls_conf = cls_conf[mask]
            bbox_xywhs.append(bbox_xywh)
            cls_confs.append(cls_conf)
            ims.append(im)

        # do tracking
        outputs = self.deepsort.update(bbox_xywhs, cls_confs, ims, tracking_target=None)

        # if detection present draw bounding boxes
        if len(outputs) > 0:
            bbox_tlwh = []
            self.bbox_xyxy = outputs[:, :4]
            # detection indices
            self.identities = outputs[:, -1]
            ori_im = draw_boxes(ori_im, self.bbox_xyxy, self.identities)

            for bb_xyxy in self.bbox_xyxy:
                bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

        # draw frame count
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 500)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        frame_count = ("Frame no: %d" % self.idx_frame)
        cv2.putText(ori_im, frame_count,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        # draw tracking number
        if self.idx_tracked:
            tracking_str = ("Tracking: %d" % self.idx_tracked)
        else:
            tracking_str = ("Tracking: None")

        bottomLeftCornerOfText = (10, 550)
        cv2.putText(ori_im, tracking_str,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        # Create CompressedImage
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', ori_im)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)

        """publishing to topics"""
        msg = self.to_spencer_msg(outputs, cls_conf, points, im.shape)
        msg.header = points.header
        self.spencer_pub.publish(msg)


def main():
    '''Initializes and cleanup ros node'''
    rospy.init_node('person_reid', anonymous=True)
    _ = VideoTracker()
    rospy.spin()


if __name__ == "__main__":
    main()
