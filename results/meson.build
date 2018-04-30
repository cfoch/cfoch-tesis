project('gst-plugins-cheese-stats', 'c',
        version : '1.13.0.1',
        meson_version : '>= 0.40.1')

gst_version = meson.project_version()
version_arr = gst_version.split('.')
gst_version_major = version_arr[0]
gst_version_minor = version_arr[1]
gst_version_micro = version_arr[2]
if version_arr.length() == 4
  gst_version_nano = version_arr[3]
else
  gst_version_nano = 0
endif
gst_req = '>= @0@.@1@.0'.format(gst_version_major, gst_version_minor)

gst_dep = dependency('gstreamer-1.0', version : gst_req,
  fallback : ['gstreamer', 'gst_dep'], required : true)
graphene_dep = dependency('graphene-1.0', required : true)
gstcheese_dep = dependency('gstcheese-1.0', required : true)

exe = executable('measure_cheesefacetrack',
  'measure_cheesefacetrack.c',
  install : false,
  dependencies : [gst_dep, graphene_dep, gstcheese_dep]
)
