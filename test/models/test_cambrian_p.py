from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from lmms_eval.models.chat import cambrian_p
from lmms_eval.models.chat.cambrian_p import CambrianP


class _ImageProcessor:
    image_mean = (0.5, 0.5, 0.5)

    def __init__(self):
        self.images = []

    def preprocess(self, images, return_tensors):
        assert return_tensors == "pt"
        self.images.append(images)
        return {"pixel_values": torch.zeros(len(images), 3, 384, 384)}


def _bare_cambrian(*, use_camera_tokens):
    model = object.__new__(CambrianP)
    model._config = SimpleNamespace(use_camera_tokens=use_camera_tokens)
    model._image_processor = _ImageProcessor()
    model._normalise_image = lambda image: image
    return model


class TestCambrianPMediaProcessing(unittest.TestCase):
    def test_camera_token_image_is_processed_as_exactly_one_frame(self):
        model = _bare_cambrian(use_camera_tokens=True)
        image = Image.new("RGB", (448, 448))
        with (
            patch.object(cambrian_p, "expand2square", side_effect=lambda frame, _background: frame),
            patch.object(cambrian_p, "process_images", side_effect=AssertionError("anyres image path was used")),
        ):
            tensors, sizes = model._process_media([image])

        self.assertEqual(len(tensors), 1)
        self.assertEqual(tensors[0].shape, (1, 3, 384, 384))
        self.assertEqual(sizes, [(448, 448)])

    def test_image_without_camera_tokens_keeps_anyres_processing(self):
        model = _bare_cambrian(use_camera_tokens=False)
        image = Image.new("RGB", (448, 448))
        anyres = torch.zeros(1, 5, 3, 384, 384)
        with patch.object(cambrian_p, "process_images", return_value=anyres):
            tensors, sizes = model._process_media([image])

        self.assertIs(tensors, anyres)
        self.assertEqual(sizes, [(448, 448)])

    def test_video_keeps_existing_prefix_processing(self):
        model = _bare_cambrian(use_camera_tokens=True)
        prefix = [torch.zeros(12, 3, 384, 384)]
        with patch.object(cambrian_p, "process_videos", return_value=(prefix, [[448, 448]], None)) as process:
            tensors, sizes = model._process_media(["prefix.mp4"])

        self.assertIs(tensors, prefix)
        self.assertEqual(sizes, [[448, 448]])
        process.assert_called_once_with(["prefix.mp4"], model._image_processor, model._config, num_threads=-1)


if __name__ == "__main__":
    unittest.main()
