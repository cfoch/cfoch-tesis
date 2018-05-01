# -*- coding: utf-8 -*-
# cfoch-tesis results
# Copyright (c) 2018 Fabian Orccon <cfoch.fabian@gmail.com>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the
# Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle

from random import randint
from rectangle import Rectangle
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
from IPython import embed


def get_centroid(bounding_box):
    return tuple((np.array(bounding_box[0]) + np.array(bounding_box[1])) / 2)

def random_color():
    return (randint(0, 255), randint(0, 255), randint(0, 255))

def euclidean_distance(pt1, pt2):
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

def map_frames(lines):
    frames = {}
    frames_faces_counter = {}
    n_faces = 0
    for line in lines:
        tokens = line.split()
        frame_number = int(tokens[0].split("Frame")[1].split(":")[0])
        bounding_box = [int(i) for i in tokens[1:]]
        bounding_box[2] = bounding_box[0] + bounding_box[2]
        bounding_box[3] = bounding_box[1] + bounding_box[3]
        bounding_box = [
            (bounding_box[0], bounding_box[1]),
            (bounding_box[2], bounding_box[3])
        ]
        if not frame_number in frames:
            frames[frame_number] = {}

        if not frame_number in frames_faces_counter:
            frames_faces_counter[frame_number] = 1
        else:
            frames_faces_counter[frame_number] += 1

        face_id = frames_faces_counter[frame_number] - 1
        frames[frame_number][face_id] = bounding_box
        n_faces = max(frames_faces_counter[frame_number], n_faces)
    return frames, n_faces

def map_frames_to_list(map_frames):
    frames = []
    n_frames = max(map_frames.keys()) + 1
    for n_frame in range(n_frames):
        if not n_frame in map_frames:
            frames.append(None)
        else:
            frames.append(map_frames[n_frame])
    return frames

def last_frame(frames, n_real_faces):
    reversed_frames = frames[::-1]
    for frame in reversed_frames:
        if len(frame) == n_real_faces:
            return frame
    return None

def reorder_frames(frames_in, n_faces, n_real_faces=2):
    first_frame_is_read = False
    frames = []
    for frame_number in range(len(frames_in)):
        frame = frames_in[frame_number]
        if frame is None:
            frames.add(None)
            continue

        if first_frame_is_read:
            lst_frame = last_frame(frames, n_real_faces)
            prev_ids = list(lst_frame.keys())
            cur_ids = list(frame.keys())

            prev_centroids = {}
            for prev_id in prev_ids:
                bounding_box = lst_frame[prev_id]
                centroid = get_centroid(bounding_box)
                prev_centroids[prev_id] = get_centroid(bounding_box)
                # prev_centroids.append(centroid)

            cur_centroids = {}
            for cur_id in cur_ids:
                bounding_box = frame[cur_id]
                centroid = get_centroid(bounding_box)
                cur_centroids[cur_id] = get_centroid(bounding_box)
                # cur_centroids.append(centroid)

            cost_matrix = []
            for cur_id in cur_ids:
                cur_centroid = cur_centroids[cur_id]
                row = []
                for prev_id in prev_ids:
                    prev_centroid = prev_centroids[prev_id]
                    cost = euclidean_distance(prev_centroid, cur_centroid)
                    row.append(cost)
                cost_matrix.append(row)

            cost_matrix = np.array(cost_matrix)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            new_frame = {}
            for i_cur, j_prev in zip(row_ind, col_ind):
                cur_id = cur_ids[i_cur]
                prev_id = prev_ids[j_prev]
                try:
                    # new_frame[cur_id] = frame[prev_id]
                    new_frame[prev_id] = frame[cur_id]
                except KeyError:
                    continue
            frames.append(new_frame)

        if not first_frame_is_read:
            first_frame_is_read = frame is not None
            frames.append(frame)
    return frames

def reorder_frames_faces_ids(frames):
    first_frame = frames[0]
    first_frame_tuples =\
        [(first_frame[face_id][0], face_id) for face_id in first_frame]
    first_frame_sorted = sorted(first_frame_tuples)

    faces_ids_map = {}
    for i, t in enumerate(first_frame_sorted):
        face_id = t[1]
        faces_ids_map[face_id] = i

    new_frames = []
    for frame in frames:
        new_frame = {}
        for face_id in faces_ids_map:
            try:
                new_frame[face_id] = frame[faces_ids_map[face_id]]
            except KeyError:
                continue
        new_frames.append(new_frame)
    return new_frames

def measure(dataset_frames, tracker_frames, n_faces):
    res = []
    for face_id in range(n_faces):
        res.append({"recall": [], "precision": [], "f1-score": []})

    for frame_number in range(len(dataset_frames)):
        dataset_frame = dataset_frames[frame_number]
        tracker_frame = tracker_frames[frame_number]
        for face_id in range(n_faces):
            if not face_id in dataset_frame and not face_id in tracker_frame:
                # Face not detected neither in dataset nor detected by tracker.
                recall = 1.0
                precision = 1.0
                f1_score = 1.0
            elif not face_id in dataset_frame and face_id in tracker_frame:
                # Face not in dataset but detected by tracker.
                recall = 0.0
                precision = 0.0
                f1_score = 0.0
            elif face_id in dataset_frame and not face_id in tracker_frame:
                # Face in dataset but not detected by tracker.
                recall = 0.0
                precision = 0.0
                f1_score = 0.0
            else:
                dataset_bounding_box = dataset_frame[face_id]
                tracker_bounding_box = tracker_frame[face_id]
                dataset_rectangle = Rectangle(
                    *(dataset_bounding_box[0] + dataset_bounding_box[1]))
                tracker_rectangle = Rectangle(
                    *(tracker_bounding_box[0] + tracker_bounding_box[1]))
                intersection_rectangle = dataset_rectangle & tracker_rectangle
                if intersection_rectangle is None:
                    recall = 0.0
                    precision = 0.0
                    f1_score = 0.0
                else:
                    intersection_area = intersection_rectangle.area
                    recall = intersection_area / dataset_rectangle.area
                    precision = intersection_area / tracker_rectangle.area
                    f1_score = 2.0 / ((1.0 / recall) + (1.0 / precision))
            res[face_id]["recall"].append(recall)
            res[face_id]["f1-score"].append(f1_score)
            res[face_id]["precision"].append(precision)
    return res

def plot_measure(measure_result, face_id):
    for measure_type in ("precision", "recall", "f1-score"):
        data = measure_result[face_id][measure_type]
        plt.plot(data)
        print("Face %d - %s(avg): %s" % (face_id, measure_type, np.mean(data)))
        plt.title("Rostro %d: %s" % (face_id + 1, measure_type))
        plt.ylabel(measure_type)
        plt.xlabel("Cuadro (frame)")
        plt.show()

VIDEO_ASSETS_PATH = "/home/cfoch/Documents/cursos/Proyecto de Tesis 1/tesis/" \
    "datasets/Multiple_faces/Head"

dataset_filepath = os.path.join(VIDEO_ASSETS_PATH, "info_multiple_head9.txt")
video_filepath = os.path.join(VIDEO_ASSETS_PATH, "multiple_head9.mp4")


parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video",
    help="The path to the video file", required=True)
parser.add_argument("-d", "--dataset",
    help="The path to the text file with the data per face and per frame",
    required=True)
parser.add_argument("-n", "--nfaces",
    help="The number of faces in the scene", required=False)
parser.add_argument("-s", "--sort",
    help="Sort faces using the Hungarian Algorithm", action='store_true',
    required=False)
parser.add_argument("-u", "--deserialize",
    help="The file path to the file with the faces info deserialized/unpickled",
    required=False)
parser.add_argument("-t", "--tail-remove",
    help="Ignore t last lines from the dataset.",
    required=False)

args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
dataset_file = open(args.dataset, "r")
lines = [l.rstrip("\n") for l in dataset_file.readlines()]
if args.tail_remove:
    lines = lines[:-int(args.tail_remove)]

map_frames, n_faces = map_frames(lines)
frames = map_frames_to_list(map_frames)
colors = [random_color() for _ in range(n_faces)]
unsorted_frames = frames
frames = reorder_frames_faces_ids(frames)

if args.sort:
    if args.nfaces is not None:
        frames = reorder_frames(frames, n_faces, args.nfaces)
    else:
        frames = reorder_frames(frames, n_faces)

tracker_frames = None
measure_result = None
if args.deserialize is not None:
    pickle_f = open(args.deserialize, "rb")
    tracker_frames = pickle.load(pickle_f)

    tracker_frames = reorder_frames_faces_ids(tracker_frames)

    measure_result = measure(frames, tracker_frames, n_faces)


    for face_id in range(n_faces):
        plot_measure(measure_result, face_id)

frame_number = 0
while True:
    ret, img = cap.read()
    if not ret:
        break

    print("frame number: ", frame_number)
    if frame_number < len(frames):
        frame = frames[frame_number]
        unsorted_frame = unsorted_frames[frame_number]

        for face_id in unsorted_frame:
            bounding_box = unsorted_frame[face_id]
            cv2.rectangle(img, bounding_box[0], bounding_box[1],
                (0, 96, 0), 12)

        for face_id in frame:
            bounding_box = frame[face_id]
            cv2.rectangle(img, bounding_box[0], bounding_box[1],
                colors[face_id], 3)

        if tracker_frames is not None:
            track_frame = tracker_frames[frame_number]
            for face_id in track_frame:
                bounding_box = track_frame[face_id]
                cv2.rectangle(img, bounding_box[0], bounding_box[1],
                    (0, 0, 255), 1)

    frame_number += 1
    cv2.imshow('frame', img)
    #cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
