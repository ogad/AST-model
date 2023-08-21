# Diameter measurement
# Oliver Driver
# 11/07/2023
import numpy as np
from tqdm.autonotebook import tqdm

#TODO: this has become a mess... split this out so that it's easier for IntensityField and AmplitudeField measurements.
def measure_diameters(detection: "ImagedRegion|DetectorRun|IntensityField|AmplitudeField", spec, force_nominsep=False, **kwargs):
    if hasattr(detection, "get_frames_to_measure"): #ImagedRegion or DetectorRun
        frames = detection.get_frames_to_measure(spec, **kwargs)
        xlims = detection.xlims
    elif hasattr(detection, "frames"): #IntensityField
        frames = [((istart, istart), field) for istart, field in detection.frames()]
        xlims = (0, detection.field.shape[0])
    else: # AmplitudeField
        frames = [((istart, istart), field) for istart, field in detection.intensity.frames()]
        xlims = (0, detection.intensity.field.shape[0])

    if len(frames) == 0:
        return {}
    
    # filter frames by spec.filters
    frames = [frame for frame in frames if np.all([image_filter(frame[1]) for image_filter in spec.filters])]

    frames.sort(key=lambda x: x[0][0])
    to_remove = []

    if spec.min_sep is not None and not force_nominsep:
        for i, ((ymin, ymax), frame) in enumerate(frames):
                if ymin == ymax:
                    raise ValueError("Frame has no height; likely using min_sep with a non-DectorRun object. This is unimplemented.")

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
        frame_detections = {(xlims[0]*1e6 + x_frame, ylims[0]*1e6 + y_frame): diameter for (x_frame, y_frame), diameter in frame_detections.items()}
        detected_particles = detected_particles | frame_detections

    diameters = list(detected_particles.values())
    return detected_particles