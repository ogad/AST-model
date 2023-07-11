# Diameter measurement
# Oliver Driver
# 11/07/2023

from __future__ import annotations

def measure_diameters(detection, spec, **kwargs):
    frames = detection.get_frames_to_measure(spec, **kwargs)

    frames.sort(key=lambda x: x[0][0])
    to_remove = []

    if spec.min_sep is not None:
        for i, ((ymin, ymax), frame) in enumerate(frames):
                if i == 0:
                    continue
                if ymin - frames[i-1][0][1] < spec.min_sep:
                    # mark for removal
                    to_remove.append(i)
                    to_remove.append(i-1)
        # remove duplicates
        to_remove = list(set(to_remove))
        for i in sorted(to_remove, reverse=True):
            del frames[i]

    kwargs["bounded"] = spec.bound
    kwargs["filled"] = spec.filled

    detected_particles = {}
    for ylims, frame_intensity in frames:
        frame_detections = frame_intensity.measure_diameters(diameter_method=spec.diameter_method, **kwargs)
        # transform keys to global coordinates
        frame_detections = {(detection.xlims[0]*1e6 + x_frame, ylims[0]*1e6 + y_frame): diameter for (x_frame, y_frame), diameter in frame_detections.items()}
        detected_particles = detected_particles | frame_detections

    diameters = list(detected_particles.values())
    return detected_particles