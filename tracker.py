from deep_sort_realtime.deepsort_tracker import DeepSort
import config
import time

class HornetTracker:
    def __init__(self):
        self.deepsort = DeepSort(max_age=30, n_init=config.MIN_HITS, nn_budget=100)
        self.track_history = {}
        self.current_hornets = set()

    def update(self, detections, frame):
        tracks = self.deepsort.update_tracks(detections, frame=frame)
        current_time = time.time()
        self.current_hornets.clear()

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            track_id = track.track_id
            if track_id not in self.track_history:
                self.track_history[track_id] = {'start_time': current_time, 'detected': False}

            if current_time - self.track_history[track_id]['start_time'] >= config.DETECTION_PERSISTENCE:
                if not self.track_history[track_id]['detected']:
                    self.track_history[track_id]['detected'] = True
                self.current_hornets.add(track_id)

        return tracks