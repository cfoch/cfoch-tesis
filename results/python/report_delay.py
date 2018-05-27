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
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import subprocess
import sys
from statistics import mean


def plot_latency_data(data):
    plt.plot(data * 1e-9)
    plt.title("Latencia promedio por Buffer")
    plt.ylabel("Latencia promedio (segundos)")
    plt.xlabel("Buffer")
    plt.show()


def find_binary_path(cmd):
    proc = subprocess.Popen(["/usr/bin/which", cmd],
                            env=os.environ.copy(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = proc.stdout.read()
    output = output.split()[0].decode("utf-8")
    return output


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--landmark",
    help="The path to the landmark shape model", required=False)
parser.add_argument("-o", "--sprite",
    help="The path to the sprite", required=False)
parser.add_argument("-n", "--times",
    help="The number of times to execute gst-launch-1.0", required=True)
parser.add_argument("-d", "--display",
    help="Sets whether to display or not a plot of the latency.",
    action="store_true", required=False)
parser.add_argument("-m", "--mean",
    help="Sets whether to display or not the mean.",
    action="store_true", required=False)
parser.add_argument("-s", "--std",
    help="Sets whether to display or not the standard deviation.",
    action="store_true", required=False)
args = parser.parse_args()

GST_LAUNCH = find_binary_path("gst-launch-1.0")
GST_TRACE_FLAGS = "GST_DEBUG='GST_TRACER:7' GST_TRACERS=latency"
PIPELINE_TMPL =\
    "v4l2src ! videoconvert ! video/x-raw,framerate=30/1 ! "\
    "cheesefacetrack max-distance-factor=0.15 scale-factor=0.5 %s"\
    "   display-landmark=true detection-gap-duration=10 ! "\
    "fakesink num-buffers=500"
PIPELINE_SPRITE_TMPL =\
    "v4l2src ! videoconvert ! video/x-raw,framerate=30/1 ! "\
    "cheesefacetrack max-distance-factor=0.15 scale-factor=0.5 %s"\
    "   display-landmark=true detection-gap-duration=10 ! videoconvert ! "\
    "cheesefaceoverlay location='%s' ! fakesink num-buffers=500"

PIPELINE_NO_LANDMARK = PIPELINE_TMPL % ""
PIPELINE_LANDMARK = PIPELINE_TMPL % ("landmark='%s'" % args.landmark)
PIPELINE_SPRITE = PIPELINE_SPRITE_TMPL % ("landmark='%s'" % args.landmark,
                                          args.sprite)
CMD_NO_LANDMARK = "%s %s %s" % (GST_TRACE_FLAGS, GST_LAUNCH,
                                PIPELINE_NO_LANDMARK)
CMD_LANDMARK = "%s %s %s" % (GST_TRACE_FLAGS, GST_LAUNCH, PIPELINE_LANDMARK)
CMD_SPRITE = "%s %s %s" % (GST_TRACE_FLAGS, GST_LAUNCH, PIPELINE_SPRITE)

if args.landmark is None:
    cmd = CMD_NO_LANDMARK
else:
    if args.sprite is not None:
        cmd = CMD_SPRITE
    else:
        cmd = CMD_LANDMARK

print(cmd)


env = os.environ.copy()
# env.update(ENV)

# print(env)
# subprocess.Popen(cmd, shell=True, env=ENV)
# proc = subprocess.Popen([GST_LAUNCH, PIPELINE_NO_LANDMARK], env=ENV)

# cmd_args = [find_binary_path("gst-inspect-1.0"), "cheesefacetrack"]
# proc = subprocess.Popen(cmd_args, env=ENV)


# proc = subprocess.Popen([GST_LAUNCH, PIPELINE_NO_LANDMARK], env=env)

latency_datalist = []

print("Measuring whole pipeline latency")
for i in range(int(args.times)):
    print("Execution %d... " % (i + 1), end="", file=sys.stderr)
    sys.stdout.flush()
    proc = subprocess.Popen(cmd, shell=True, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate()

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(err)
        tmp_file.close()
        proc = subprocess.Popen("python measure_delay.py -i %s" % tmp_file.name,
                                shell=True, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, _ = proc.communicate()
        latency_data = []
        for i, line in enumerate(out.split()):
            latency_data.append(int(line))
        # latency_data = [int(line) for line in out.split()]
        latency_datalist.append(latency_data)
    print("Ready.", file=sys.stderr)

latency_avg = list(map(mean, zip(*latency_datalist)))
for latency in latency_avg:
    print(latency)

if args.mean:
    print("mean: ", np.mean(latency_avg))
if args.std:
    print("std: ", np.std(latency_avg))
if args.display:
    plot_latency_data(np.array(latency_avg))
