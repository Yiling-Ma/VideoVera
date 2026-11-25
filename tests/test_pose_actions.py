import copy
import math
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from videovera.data.navigation.actions import UncertainRoute, extract_route, file_sha256
from videovera.data.navigation.questions import build_record, choices_for, choices_from_training_pool


def trajectory(actions):
    frames = []
    x, z, yaw = 0., 0., 0.

    def emit():
        angle = math.radians(yaw)/2
        i = len(frames)
        frames.append(dict(frame_index=i, timestamp_sec=i/10,
                           position_xyz_m=[x, 0., z], rotation_xyzw=[0., math.sin(angle), 0., math.cos(angle)]))

    for _ in range(3):
        emit()
    for kind, value in actions:
        if kind == "go":
            count = round(value/.1)
            for _ in range(count):
                x -= math.sin(math.radians(yaw))*.1
                z -= math.cos(math.radians(yaw))*.1
                emit()
        elif kind == "turn":
            count = max(1, round(abs(value)/15))
            for _ in range(count):
                yaw += value/count
                emit()
        for _ in range(3):
            emit()
    return dict(video={"fps": 10, "frame_count": len(frames)}, camera_trajectory={
        "basis": "measured_camera_pose", "capture_source": "synthetic test fixture only",
        "coordinate_system": "habitat_y_up_camera_minus_z_xyzw",
        "video_sha256": "0"*64, "frames": frames})


class PoseRouteTests(unittest.TestCase):
    def test_training_negatives_come_only_from_same_length_pool(self):
        pool = {'Turn left. Turn right.':20, 'Turn right. Turn left.':15,
                'Turn left. Turn left.':10, 'Turn right. Turn right.':8}
        result = choices_from_training_pool('training-pool', ['TURN_LEFT','TURN_RIGHT'], pool)
        choices, key = result
        self.assertEqual(set(choices.values()), set(pool))
        self.assertEqual(choices[key], 'Turn left. Turn right.')
        self.assertEqual(result, choices_from_training_pool('training-pool', ['TURN_LEFT','TURN_RIGHT'], pool))
        with self.assertRaises(ValueError):
            choices_from_training_pool('bad-pool', ['TURN_LEFT','TURN_RIGHT'], dict(pool, **{'Turn left.':3}))

    def test_reverse_does_not_reveal_answer_via_turn_back_prefix(self):
        actions = ["TURN_BACK", "GO_STRAIGHT", "TURN_LEFT", "GO_STRAIGHT"]
        choices, _ = choices_for("reverse-prefix", actions, "reverse_route_plan")
        self.assertEqual(len(choices), 4)
        self.assertTrue(all(s.startswith("Turn back.") for s in choices.values()))

    def test_multi_turn_options_have_no_unique_hamming_center(self):
        for turns in range(2, 7):
            for seed in range(20):
                actions = ["TURN_BACK", "GO_STRAIGHT"] + ["TURN_LEFT", "GO_STRAIGHT"] * (turns-1)
                choices, key = choices_for(str(seed), actions)
                seqs = [s.split(". ") for s in choices.values()]
                profiles = [sorted(sum(a != b for a, b in zip(s, t)) for t in seqs) for s in seqs]
                self.assertTrue(all(p == profiles[0] for p in profiles))
                self.assertEqual(len(set(choices.values())), 4)
                self.assertEqual(choices[key].split(". ")[0], "Turn back")
                self.assertEqual((choices, key), choices_for(str(seed), actions))

    def test_choice_actions_keep_fixed_length(self):
        for actions in (["TURN_LEFT"], ["TURN_LEFT", "TURN_RIGHT"],
                        ["TURN_BACK", "GO_STRAIGHT"]):
            choices, key = choices_for("fixed-length-case", list(actions))
            expected_turns = sum(action.startswith("TURN") for action in actions)
            self.assertEqual(choices[key], " ".join(
                {"TURN_LEFT": "Turn left.", "TURN_RIGHT": "Turn right.",
                 "TURN_BACK": "Turn back.", "GO_STRAIGHT": "Go straight."}[action]
                for action in actions
            ))
            self.assertTrue(all(
                sum(text.lower().count(word) for word in ("turn left", "turn right", "turn back"))
                == expected_turns
                for text in choices.values()
            ))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
    def test_real_mp4_binding_and_question_generation(self):
        payload = trajectory([("go", 3), ("turn", 90)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "route.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x64:r=10",
                            "-frames:v", str(payload["video"]["frame_count"]), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", str(video)], check=True)
            payload["camera_trajectory"]["video_sha256"] = file_sha256(video)
            self.assertEqual(extract_route(payload, video)["reverse_action_sequence"], ["TURN_LEFT", "GO_STRAIGHT"])
            annotation = root / "route.actions.json"
            annotation.write_text(json.dumps(payload))
            sample = root / "sample.jsonl"
            sample.write_text(json.dumps(dict(id="test", task_type="reverse_route_plan", dataset_source="r2r",
                                             scene_id="scene", video_id="route", video_path=str(video),
                                             annotation_path=str(annotation)))+"\n")
            row = build_record(payload, video, annotation, "scene", "route", "reverse_route_plan", "test")
            self.assertEqual(row["answer"], "Turn left. Go straight.")
            wrong = copy.deepcopy(payload)
            wrong["camera_trajectory"]["video_sha256"] = "0"*64
            with self.assertRaisesRegex(UncertainRoute, "hash_mismatch"):
                extract_route(wrong, video)
            wrong = copy.deepcopy(payload)
            wrong["video"]["fps"] = 20
            with self.assertRaisesRegex(UncertainRoute, "actual_video_pose"):
                extract_route(wrong, video)

    def test_straight_with_terminal_left_returns_left_not_back(self):
        route = extract_route(trajectory([("go", 3), ("turn", 90)]))
        self.assertEqual(route["forward_actions"], ["GO_STRAIGHT", "TURN_LEFT"])
        self.assertEqual(route["reverse_action_sequence"], ["TURN_LEFT", "GO_STRAIGHT"])

    def test_terminal_right_and_back(self):
        for angle, initial in [(-90, "TURN_RIGHT"), (0, "TURN_BACK"), (180, None)]:
            with self.subTest(angle=angle):
                route = extract_route(trajectory([("go", 3), ("turn", angle)]))
                expected = ([initial] if initial else []) + ["GO_STRAIGHT"]
                self.assertEqual(route["reverse_action_sequence"], expected)

    def test_zero_net_turn_does_not_erase_actual_zigzag(self):
        route = extract_route(trajectory([("go", 2), ("turn", 90), ("go", 2), ("turn", -90), ("go", 2)]))
        self.assertEqual(route["forward_actions"], ["GO_STRAIGHT", "TURN_LEFT", "GO_STRAIGHT", "TURN_RIGHT", "GO_STRAIGHT"])
        self.assertEqual(route["reverse_action_sequence"], ["TURN_BACK", "GO_STRAIGHT", "TURN_LEFT", "GO_STRAIGHT", "TURN_RIGHT", "GO_STRAIGHT"])

    def test_long_route_is_not_capped_at_seven_actions(self):
        actions = [("go", 2)]
        for i in range(6):
            actions.extend([("turn", 90 if i % 2 == 0 else -90), ("go", 2)])
        route = extract_route(trajectory(actions))
        self.assertEqual(len(route["forward_actions"]), 13)
        self.assertEqual(len(route["reverse_action_sequence"]), 14)

    def test_lateral_jitter_and_camera_wobble_merge(self):
        payload = trajectory([("go", 4)])
        for frame in payload["camera_trajectory"]["frames"][4:-4]:
            i = frame["frame_index"]
            frame["position_xyz_m"][0] += .06*math.sin(i)
            yaw = math.radians(8*math.sin(i))/2
            frame["rotation_xyzw"] = [0., math.sin(yaw), 0., math.cos(yaw)]
        self.assertEqual(extract_route(payload)["forward_actions"], ["GO_STRAIGHT"])

    def test_minor_course_correction_is_one_straight_leg(self):
        route = extract_route(trajectory([("go", 2), ("turn", 20), ("go", 2)]))
        self.assertEqual(route["forward_actions"], ["GO_STRAIGHT"])
        self.assertEqual(len(route["legs"]), 1)

    def test_initial_turn_not_repeated_at_return_position(self):
        route = extract_route(trajectory([("turn", -90), ("go", 3)]))
        self.assertEqual(route["forward_actions"], ["TURN_RIGHT", "GO_STRAIGHT"])
        self.assertEqual(route["reverse_action_sequence"], ["TURN_BACK", "GO_STRAIGHT"])

    def test_no_pose_or_reference_only_is_rejected(self):
        for payload in [{}, {"reference_path": [{"position_xyz_m": [0, 0, 0]}, {"position_xyz_m": [0, 0, -2]}]},
                        {"janus_raw_action_sequence": ["MOVE_FORWARD"]*10}]:
            with self.assertRaisesRegex(UncertainRoute, "missing_measured"):
                extract_route(payload)

    def test_bad_timestamp_and_jump_rejected(self):
        payload = trajectory([("go", 3)])
        payload["camera_trajectory"]["frames"][8]["timestamp_sec"] += .1
        with self.assertRaisesRegex(UncertainRoute, "timestamp"):
            extract_route(payload)
        payload = trajectory([("go", 3)])
        payload["camera_trajectory"]["frames"][8]["position_xyz_m"][0] += 2
        with self.assertRaisesRegex(UncertainRoute, "jump"):
            extract_route(payload)

    def test_short_outgoing_leg_is_not_invented(self):
        payload = trajectory([("go", 3), ("turn", 90), ("go", .3)])
        with self.assertRaisesRegex(UncertainRoute, "short_or_ambiguous"):
            extract_route(payload)

    def test_no_stable_endpoint_is_rejected(self):
        payload = trajectory([("go", 3), ("turn", 90)])
        frames = payload["camera_trajectory"]["frames"][:-3]
        payload["camera_trajectory"]["frames"] = frames
        payload["video"]["frame_count"] = len(frames)
        # Fifteen degree steps are just within spread; increase the last swing.
        frames[-1]["rotation_xyzw"] = [0, 1, 0, 0]
        with self.assertRaisesRegex(UncertainRoute, "unstable"):
            extract_route(payload)

    def test_atomic_gold_and_answer_rebuilt(self):
        payload = trajectory([("go", 3), ("turn", 90)])
        with patch("videovera.data.navigation.actions.file_sha256", return_value="0"*64), patch(
            "videovera.data.navigation.actions.video_timestamps", return_value=[f["timestamp_sec"] for f in payload["camera_trajectory"]["frames"]]
        ):
            row = build_record(payload, "v.mp4", "v.actions.json", "scene", "v", "reverse_route_plan")
        self.assertEqual(row["answer"], "Turn left. Go straight.")
        self.assertEqual(row["choices"][row["answer_key"]], row["answer"])
        forward = [s for s in row["gold_trace"] if s["claim_type"] == "forward"]
        self.assertEqual(len(forward), 2)
        self.assertFalse(row["metadata"]["evaluation_valid"])

    def test_landmarks_must_be_reanchored_and_variable_length(self):
        payload = trajectory([("go", 2), ("turn", 90), ("go", 2), ("turn", -90), ("go", 2), ("turn", 90), ("go", 2)])
        route = extract_route(payload)
        with patch("videovera.data.navigation.actions.file_sha256", return_value="0"*64), patch(
            "videovera.data.navigation.actions.video_timestamps", return_value=[f["timestamp_sec"] for f in payload["camera_trajectory"]["frames"]]
        ):
            with self.assertRaisesRegex(UncertainRoute, "reanchoring"):
                build_record(payload, "v.mp4", "v.json", "s", "v", "route_plan")
            payload["route_plan_landmarks"] = {"pose_signature": route["pose_signature"], "anchors": [
                {"waypoint_index": b["waypoint_index"], "canonical_video_frame": b["frame_index"],
                 "selected_landmark": {"label": f"object {i}", "usable_for_route_plan": True}}
                for i, b in enumerate(route["boundaries"])]}
            row = build_record(payload, "v.mp4", "v.json", "s", "v", "route_plan")
        self.assertEqual(row["question"].count("[?]"), 3)
        self.assertEqual(len([s for s in row["gold_trace"] if s["claim_type"] == "forward"]), 7)


if __name__ == "__main__":
    unittest.main()
