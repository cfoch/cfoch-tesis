/*
 * cfoch-tesis results
 * Copyright (C) 2018 Fabian Orccon <cfoch.fabian@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */
#include <gst/gst.h>
#include <gst/cheese/face/cheesemultifacemeta.h>
#include <gst/cheese/face/cheesemultifaceinfo.h>
#include <gst/cheese/face/cheesefaceinfo.h>
#include <glib.h>

#define SPIPELINE  "filesrc name=source ! decodebin ! videoconvert ! "\
                   "cheesefacetrack name=face_track ! fakesink name=sink"

static GMainLoop *loop;
static int frame_number = 0;

static void
bus_cb (GstBus *bus, GstMessage *message, gpointer user_data)
{
  GstElement *pipeline = GST_ELEMENT (user_data);

  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_ERROR:
      g_print ("we received an error!\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_EOS:
      g_main_loop_quit (loop);
      break;
    default:
      break;
  }
}

static GstPadProbeReturn
probe_cb (GstPad * pad, GstPadProbeInfo * info, gpointer user_data)
{
  if (GST_PAD_PROBE_INFO_TYPE (info) & GST_PAD_PROBE_TYPE_BUFFER) {
    GstBuffer *buf = gst_pad_probe_info_get_buffer (info);
    GstCheeseMultifaceMeta *meta = (GstCheeseMultifaceMeta *)
        gst_buffer_get_meta (buf, GST_CHEESE_MULTIFACE_META_API_TYPE);
    if (meta) {
      GstCheeseMultifaceInfoIter itr;
      GstCheeseFaceInfo *face_info;
      guint face_id;

      gst_cheese_multiface_info_iter_init (&itr, meta->faces);
      while (gst_cheese_multiface_info_iter_next (&itr, &face_id, &face_info)) {
        graphene_rect_t bounding_box;
        bounding_box = cheese_face_info_get_bounding_box (face_info);
        g_print ("%d %d %d %d %d %d\n", frame_number, face_id,
            (gint) graphene_rect_get_x (&bounding_box),
            (gint) graphene_rect_get_x (&bounding_box),
            (gint) graphene_rect_get_height (&bounding_box),
            (gint) graphene_rect_get_width (&bounding_box));
      }
      frame_number++;
    }
  }
  return GST_PAD_PROBE_OK;
}

int
main (int argc, char ** argv)
{
  GError *error = NULL;
  GstElement *pipeline, *filesrc, *face_track, *sink;
  GstBus *bus;
  GstPad *sinkpad;
  guint probe_id;

  gst_init (&argc, &argv);

  loop = g_main_loop_new (NULL, FALSE);

  pipeline = gst_parse_launch (SPIPELINE, &error);

  if (error != NULL) {
    g_critical (error->message);
  }

  filesrc = gst_bin_get_by_name (GST_BIN (pipeline), "source");
  face_track = gst_bin_get_by_name (GST_BIN (pipeline), "face_track");
  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");

  g_object_set (G_OBJECT (filesrc), "location", argv[1], NULL);

  sinkpad = gst_element_get_static_pad (sink, "sink");
  probe_id = gst_pad_add_probe (sinkpad,
        GST_PAD_PROBE_TYPE_BUFFER, probe_cb, sink, NULL);

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  gst_bus_add_signal_watch (bus);
  g_signal_connect (bus, "message", (GCallback) bus_cb, pipeline);

  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  g_main_loop_run (loop);
  gst_element_set_state (pipeline, GST_STATE_NULL);

  gst_object_unref (pipeline);
  g_main_loop_unref (loop);
  return 0;
}
