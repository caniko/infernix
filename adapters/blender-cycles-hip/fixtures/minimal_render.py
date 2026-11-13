"""Fixture operation: Cycles render of the default cube to a PNG sequence.

This is the adapter's own smoke fixture, not a studio operation: it exists so
the adapter can be exercised end to end without studio-pipeline checked out.
It obeys the same environment contract as a real operation script.
"""

import os

import bpy

frames = os.environ["INFERNIX_FRAMES_DIR"]
start = int(os.environ["INFERNIX_FRAME_START"])
end = int(os.environ["INFERNIX_FRAME_END"])

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.samples = 8
scene.render.resolution_x = 160
scene.render.resolution_y = 120
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = os.path.join(frames, "")
scene.frame_start = start
scene.frame_end = end

bpy.context.scene.frame_set(start)
for frame in range(start, end + 1):
    bpy.context.scene.frame_set(frame)
    bpy.ops.render.render(write_still=False, animation=True)
    break

print("INFERNIX_FIXTURE_RENDERED", len(os.listdir(frames)))
