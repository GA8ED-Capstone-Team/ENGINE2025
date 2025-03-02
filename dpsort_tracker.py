from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortTracker:
    def __init__(self):
        self.object_tracker = DeepSort(
            max_age = 20,
        )

    def track(self, detections, frame):
        tracks = self.object_tracker.update_tracks(detections, frame = frame)

        tracking_ids = []
        boxes = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tracking_ids.append(track.track_id)
            ltrb = track.to_ltrb()
            boxes.append(ltrb)
        
        return tracking_ids, boxes


