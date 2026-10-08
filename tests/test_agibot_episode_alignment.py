import unittest

from agibot2lerobot.check_episode_alignment import count_frames_in_interval


class EpisodeAlignmentTests(unittest.TestCase):
    def test_video_segment_includes_start_and_excludes_next_episode_boundary(self):
        fps = 30
        frame_times = [index / fps for index in range(6)]

        # The frame at 3/fps belongs to the following episode, not this interval.
        self.assertEqual(count_frames_in_interval(frame_times, 0, 3 / fps), 3)

    def test_missing_video_frame_is_detected_by_segment_count(self):
        fps = 30
        frame_times = [0, 1 / fps, 3 / fps]

        self.assertEqual(count_frames_in_interval(frame_times, 0, 3 / fps), 2)


if __name__ == "__main__":
    unittest.main()
