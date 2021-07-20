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
        # initialize publishers
        self.image_pub = rospy.Publisher("~image/compressed", CompressedImage, queue_size=1)
        self.spencer_pub = rospy.Publisher("~detected_persons", DetectedPersons, queue_size=1)

        # initialize services to interact with node
        self.target_clear_srv = rospy.Service("~clear_target", clear_target, self.clear_track)
        self.target_choose_srv = rospy.Service("~choose_target", choose_target, self.select_target)

        self.deepsort = None

        image_sub = message_filters.Subscriber('~image', CompressedImage)
        points_sub = message_filters.Subscriber("~points", PointCloud2)
        bbox_sub = message_filters.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes)
        sub = message_filters.TimeSynchronizer([image_sub, points_sub, bbox_sub], 10)
        sub.registerCallback(self.ros_deepsort_callback)

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

    def to_spencer_msg(self, detections, points, shape):
        persons = DetectedPersons()
        width, height = shape[:2]
        for p in detections:
            tlwh = self.deepsort._xyxy_to_tlwh(p[:4])
            x, y = int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3] / 2)
            centers = [[xx, yy] for yy in range(max(y - 3, 0), min(y + 3, height))
                       for xx in range(max(x - 3, 0), min(x + 3, width))]
            pts = [p for p in pc2.read_points(points, ('x', 'y', 'z'), uvs=centers)]
            pt = np.mean(pts, axis=0)
            person = DetectedPerson()
            person.modality = DetectedPerson.MODALITY_GENERIC_RGBD
            person.pose.pose.position.x = pt[0]
            person.pose.pose.position.y = pt[1]
            person.pose.pose.position.z = pt[2]
            person.confidence = 0.5
            person.detection_id = p[-1]
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

        # convert ros compressed image message to opencv
        np_arr = np.fromstring(color.data, np.uint8)

        ori_im = cv2.imdecode(np_arr, flags=cv2.IMREAD_COLOR)
        im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

        # do detection
        bbox_xywh = []
        cls_conf = []
        cls_ids = []
        for b in bbox.bounding_boxes:
            bbox_xywh.append([b.x + b.w / 2, b.y + b.h / 2, b.w, b.h])
            cls_conf.append(b.prob)
            cls_ids.append(0 if 'person' in b.Class else 1)

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

        # do tracking
        outputs = self.deepsort.update(bbox_xywh, cls_conf, im, tracking_target=None)

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
        msg = self.to_spencer_msg(outputs, points, im.shape)
        msg.header = points.header
        self.spencer_pub.publish(msg)


def main():
    '''Initializes and cleanup ros node'''
    rospy.init_node('person_reid', anonymous=True)
    _ = VideoTracker()
    rospy.spin()


if __name__ == "__main__":
    main()
