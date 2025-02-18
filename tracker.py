from collections import defaultdict
import math

class Tracker:
    def __init__(self, max_distance=35, max_history=30):
        self.track_history = defaultdict(lambda: [])  # {id: [(x, y), (x, y), ...]}
        self.id_count = 0
        self.max_distance = max_distance
        self.max_history = max_history

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            same_object_detected = False
            for obj_id, track in self.track_history.items():
                prev_center = track[-1]
                dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
                if dist < self.max_distance:
                    self.track_history[obj_id].append((cx, cy))
                    if len(self.track_history[obj_id]) > self.max_history:
                        self.track_history[obj_id].pop(0)  # Retain only the last 'max_history' points
                    objects_bbs_ids.append([x1, y1, x2, y2, obj_id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.track_history[self.id_count].append((cx, cy))
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                self.id_count += 1

        new_track_history = defaultdict(lambda: [])
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            new_track_history[object_id] = self.track_history[object_id]

        self.track_history = new_track_history.copy()
        return objects_bbs_ids
