import unittest

import numpy as np
import torch
from PIL import Image

from lmms_eval.models.chat.spatial_mllm import _prepare_spatial_mllm_inputs


class TestPrepareSpatialMLLMInputs(unittest.TestCase):
    def test_last_frame_repeats_image_for_temporal_patch(self):
        image = Image.fromarray(np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3))

        batch = _prepare_spatial_mllm_inputs(
            {},
            video_inputs=None,
            image_inputs=[image],
            image_temporal_patch_size=2,
        )

        image_tchw = batch["image_tchw"][0]
        expected_chw = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        self.assertEqual(image_tchw.shape, (2, 3, 4, 5))
        self.assertTrue(torch.equal(image_tchw[0], expected_chw))
        self.assertTrue(torch.equal(image_tchw[1], expected_chw))
        self.assertIsNone(batch["video_tchw"])

    def test_prefix_keeps_video_shape_and_values(self):
        video = torch.arange(2 * 3 * 4 * 5, dtype=torch.uint8).reshape(2, 3, 4, 5)

        batch = _prepare_spatial_mllm_inputs({}, video_inputs=[video], image_inputs=None)

        video_tchw = batch["video_tchw"][0]
        self.assertEqual(video_tchw.shape, (2, 3, 4, 5))
        self.assertTrue(torch.equal(video_tchw, video.float() / 255.0))
        self.assertIsNone(batch["image_tchw"])

    def test_rejects_video_without_time_dimension(self):
        malformed_video = torch.zeros(3, 4, 5, dtype=torch.uint8)

        with self.assertRaisesRegex(ValueError, r"video tensors must have shape \(T, C, H, W\)"):
            _prepare_spatial_mllm_inputs({}, video_inputs=[malformed_video], image_inputs=None)

    def test_rejects_nonpositive_image_temporal_patch_size(self):
        with self.assertRaisesRegex(ValueError, "temporal patch size must be positive"):
            _prepare_spatial_mllm_inputs(
                {},
                video_inputs=None,
                image_inputs=[Image.new("RGB", (4, 4))],
                image_temporal_patch_size=0,
            )


if __name__ == "__main__":
    unittest.main()
