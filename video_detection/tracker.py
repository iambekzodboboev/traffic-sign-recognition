"""Stage 9.4 (part 2) -- simple IoU-based tracking across sampled frames.

Purpose: the same physical sign is detected independently in every sampled
frame as the car approaches it. Without tracking, that's reported as many
separate, possibly-conflicting guesses. This module recognizes "this is
still the same sign as before" by position overlap, and only reports a
sign once it's been confidently and *consistently* recognized -- not just
confident in one lucky frame.
"""

IOU_MATCH_THRESHOLD = 0.15   # how much two boxes must overlap to count as "the same sign"
MAX_FRAMES_MISSING = 3       # how many sampled frames a track can go unseen before it's finalized
CONFIDENCE_THRESHOLD = 0.60  # same idea as the bot's threshold

# Real-world testing found that a sign off to the side of the road can move
# across the frame faster than area-overlap can follow at a low sample rate
# -- two consecutive samples can have literally zero box overlap even though
# it's obviously the same sign, just moving. IoU alone would permanently
# lose the track there. As a fallback when IoU finds no match, also accept
# a match by center-to-center distance relative to box size, which still
# works when the box has moved but not grown/shrunk wildly.
CENTER_DISTANCE_MAX_FACTOR = 1.5


def _iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area else 0.0


def _center_distance_score(box_a, box_b):
    """Returns a 0-1 "closeness" score based on center distance relative to
    box size (1.0 = same center, 0.0 = at or beyond CENTER_DISTANCE_MAX_FACTOR
    box-widths apart). Used only when IoU finds no overlap at all."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    acx, acy = ax + aw / 2, ay + ah / 2
    bcx, bcy = bx + bw / 2, by + bh / 2
    dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    avg_size = (aw + ah + bw + bh) / 4
    if avg_size == 0:
        return 0.0
    max_dist = CENTER_DISTANCE_MAX_FACTOR * avg_size
    return max(0.0, 1.0 - dist / max_dist)


CENTER_DISTANCE_MIN_SCORE = 0.5  # required closeness when falling back (no IoU overlap at all)


def _is_match(box_a, box_b):
    """True if box_b should be considered the same physical sign as box_a."""
    if _iou(box_a, box_b) >= IOU_MATCH_THRESHOLD:
        return True
    # No area overlap at all -- fall back to center-distance, for a sign
    # that moved faster across the frame than area-overlap can follow.
    return _center_distance_score(box_a, box_b) >= CENTER_DISTANCE_MIN_SCORE


def _match_score(box_a, box_b):
    """Ranks candidates when more than one passes _is_match: a real overlap
    always outranks a center-distance-only fallback match."""
    iou = _iou(box_a, box_b)
    if iou > 0:
        return 1.0 + iou
    return _center_distance_score(box_a, box_b)


class Track:
    _next_id = 1

    def __init__(self, frame_idx, box):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.last_seen_frame = frame_idx
        self.observations = []  # list of (class_id, name, category, confidence)

    def add_observation(self, frame_idx, box, class_id, name, category, confidence):
        self.box = box
        self.last_seen_frame = frame_idx
        self.observations.append((class_id, name, category, confidence))

    def finalize(self):
        """Returns a report dict, or None if this track should never be
        reported (no consistent, confident majority class)."""
        confident = [o for o in self.observations if o[3] >= CONFIDENCE_THRESHOLD]
        if not confident:
            return None

        counts = {}
        for class_id, name, category, confidence in confident:
            counts.setdefault(class_id, []).append((name, category, confidence))

        majority_class_id = max(counts, key=lambda cid: len(counts[cid]))
        majority_votes = counts[majority_class_id]
        # Require the majority to actually be a majority of the *confident*
        # observations, not just whichever class happened to appear once --
        # this is the "consistent across frames, not one lucky frame" rule.
        if len(majority_votes) < max(2, len(confident) // 2 + 1) and len(confident) > 1:
            return None

        name, category, _ = max(majority_votes, key=lambda v: v[2])
        best_confidence = max(v[2] for v in majority_votes)
        return {
            "track_id": self.id,
            "class_id": majority_class_id,
            "name": name,
            "category": category,
            "confidence": best_confidence,
            "frames_seen": len(self.observations),
            "frames_confident": len(confident),
        }


class SignTracker:
    def __init__(self):
        self.active_tracks = []
        self.finalized_reports = []

    def update(self, frame_idx, detections):
        """detections: list of (box, class_id, name, category, confidence).
        Matches each to an active track (by box overlap) or starts a new one.
        Returns a list of track ids, one per input detection, in the same
        order -- so the caller can label each box with its stable track id."""
        # Keep original positions so we can build an aligned id list even
        # though we pop from `unmatched` as we go.
        indexed = list(enumerate(detections))
        assigned_ids = [None] * len(detections)

        for track in self.active_tracks:
            if not indexed:
                break
            best_pos, best_i, best_score = None, None, -1.0
            for pos, (orig_i, (box, *_rest)) in enumerate(indexed):
                if not _is_match(track.box, box):
                    continue
                score = _match_score(track.box, box)
                if score > best_score:
                    best_pos, best_i, best_score = pos, orig_i, score
            if best_i is not None:
                orig_i, (box, class_id, name, category, confidence) = indexed.pop(best_pos)
                track.add_observation(frame_idx, box, class_id, name, category, confidence)
                assigned_ids[orig_i] = track.id

        for orig_i, (box, class_id, name, category, confidence) in indexed:
            track = Track(frame_idx, box)
            track.add_observation(frame_idx, box, class_id, name, category, confidence)
            self.active_tracks.append(track)
            assigned_ids[orig_i] = track.id

        self._expire_stale_tracks(frame_idx)
        return assigned_ids

    def _expire_stale_tracks(self, frame_idx):
        still_active = []
        for track in self.active_tracks:
            if frame_idx - track.last_seen_frame > MAX_FRAMES_MISSING:
                report = track.finalize()
                if report is not None:
                    self.finalized_reports.append(report)
            else:
                still_active.append(track)
        self.active_tracks = still_active

    def finish(self):
        """Call once after the video ends, to finalize any tracks still active."""
        for track in self.active_tracks:
            report = track.finalize()
            if report is not None:
                self.finalized_reports.append(report)
        self.active_tracks = []
        return self.finalized_reports
