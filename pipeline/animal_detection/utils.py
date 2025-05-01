import json
import numpy as np

BEAR_CLASS = 21


def compute_stability_scores(
    json_input_path: str,
    json_output_path: str,
    stability_threshold: float,
    wild_animals: dict,
    min_frame_persistence: int = 10,
    window_size: int = 100,
    raise_alerts: bool = True,
):
    """
    Reads tracked_predictions.json, computes stability scores over
    the last `window_size` frames, writes them to json_output_path,
    and optionally prints alerts.
    """

    # Load tracked_predictions.json
    with open(json_input_path, "r") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    window = frames[-window_size:] if len(frames) > window_size else frames
    total_tracked = len(window)

    # Accumulate confidences & persistence
    object_confidences = {}
    object_persistence = {}
    for frame in window:
        seen = set()
        for tr in frame.get("tracks", []):
            cls, conf = tr["class_id"], tr.get("confidence")
            if conf is not None:
                object_confidences.setdefault(cls, []).append(conf)
            if cls not in seen:
                object_persistence[cls] = object_persistence.get(cls, 0) + 1
                seen.add(cls)

    # Compute stability scores and write to s3
    stability_scores = {}
    for cls, confs in object_confidences.items():
        pers = object_persistence.get(cls, 0)
        if pers > 0:
            score = (np.mean(confs) * pers) / total_tracked
            stability_scores[cls] = {
                "stability_score": score,
                "mean_confidence": float(np.mean(confs)),
                "persistence": pers,
            }
    with open(json_output_path, "w") as f:
        json.dump(
            {
                "video": data.get("video"),
                "total_tracked_frames": total_tracked,
                "scores": stability_scores,
            },
            f,
            indent=2,
        )
    print(f"Stability scores written to {json_output_path}")

    # Alerting
    if raise_alerts:
        for cls, info in stability_scores.items():
            score = info["stability_score"]
            mean_conf = info["mean_confidence"]
            pers = info["persistence"]
            is_wild = cls in wild_animals
            is_bear = cls == BEAR_CLASS
            if (score > stability_threshold and is_wild) or (
                is_bear and pers >= min_frame_persistence and mean_conf > 0.5
            ):
                name = wild_animals.get(cls, f"Class {cls}").upper()
                print(f"ALERT: {name} detected (score={score:.3f})")
                # TODO: Decide what needs to be done here

    return stability_scores
