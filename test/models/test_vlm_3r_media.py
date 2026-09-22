import unittest

from lmms_eval.models.model_utils.vlm_3r_media import extract_ordered_media_and_turns


class TestVLM3RMedia(unittest.TestCase):
    def test_preserves_video_then_reference_order(self):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "System prompt"}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": "clip.mp4"},
                    {"type": "image", "url": "reference.png"},
                    {"type": "text", "text": "Find this object."},
                ],
            },
        ]

        media, turns, system_prompt = extract_ordered_media_and_turns(messages, "<image>")

        self.assertEqual(media, [("video", "clip.mp4"), ("image", "reference.png")])
        self.assertEqual(turns, [("user", "<image>\n<image>\nFind this object.")])
        self.assertEqual(system_prompt, "System prompt")

    def test_preserves_reference_then_video_order(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "reference.png"},
                    {"type": "video", "url": "clip.mp4"},
                    {"type": "text", "text": "Find this object."},
                ],
            }
        ]

        media, turns, _ = extract_ordered_media_and_turns(messages, "<image>")

        self.assertEqual(media, [("image", "reference.png"), ("video", "clip.mp4")])
        self.assertEqual(turns, [("user", "<image>\n<image>\nFind this object.")])


if __name__ == "__main__":
    unittest.main()
