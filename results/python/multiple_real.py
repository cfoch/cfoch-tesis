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
import os

from random import randint
from scipy.optimize import linear_sum_assignment
from scipy import ndimage


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
        print(frame_number)
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
            print(frame_number, frame)
    return frames
        

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
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
dataset_file = open(args.dataset, "r")
lines = [l.rstrip("\n") for l in dataset_file.readlines()]

map_frames, n_faces = map_frames(lines)
frames = map_frames_to_list(map_frames)


colors = [random_color() for _ in range(n_faces)]

unsorted_frames = frames

if args.sort:
    if args.nfaces is not None:
        frames = reorder_frames(frames, n_faces, args.nfaces)
    else:
        frames = reorder_frames(frames, n_faces)

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

    frame_number += 1
    cv2.imshow('frame', img)
    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
