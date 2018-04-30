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
import pickle

from random import randint
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
from IPython import embed


def random_color():
    return (randint(0, 255), randint(0, 255), randint(0, 255))

def predataset_to_frame_map(lines):
    frames = {}
    frames_faces_counter = {}
    faces_ids = set([])
    for line in lines:
        tokens = line.split()
        frame_number = int(tokens[0])
        face_id = int(tokens[1]) - 1
        bounding_box = [int(i) for i in tokens[2:]]
        bounding_box[2] = bounding_box[0] + bounding_box[2]
        bounding_box[3] = bounding_box[1] + bounding_box[3]
        bounding_box = [
            (bounding_box[0], bounding_box[1]),
            (bounding_box[2], bounding_box[3])
        ]
        if not frame_number in frames:
            frames[frame_number] = {}
        faces_ids.add(face_id)
        frames[frame_number][face_id] = bounding_box
    n_faces = max(faces_ids) + 1
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
parser.add_argument("-m", "--serialize",
    help="The file path to the file with the faces info serialized/pickled",
    required=False)
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
dataset_file = open(args.dataset, "r")
lines = [l.rstrip("\n") for l in dataset_file.readlines()]
dataset_file.close()

frames, n_faces = predataset_to_frame_map(lines)

frames = map_frames_to_list(frames)
colors = [random_color() for _ in range(n_faces)]

if args.serialize is not None:
    pickle_f = open(args.serialize, "wb")
    pickle.dump(frames, pickle_f)
    pickle_f.close()

# embed()

frame_number = 0
while True:
    ret, img = cap.read()
    if not ret:
        break

    print("frame number: ", frame_number)
    if frame_number < len(frames):
        frame = frames[frame_number]

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
