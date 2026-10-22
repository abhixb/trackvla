import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackmania_vla.tmnf_window import (
    OpponentDialogController,
    LOAD_SCREEN_DISMISS_CLICK,
    NONE_OPPONENT_ROW,
    _configure_opponent_dialog,
    _dialog_green_groups,
    _opponent_dialog_visible,
    _show_your_ghost_enabled,
)


class OpponentDialogControllerTests(unittest.TestCase):
    def test_recognizes_dialog_title_and_selected_opponent(self):
        image = Image.new("RGB", (640, 510), (20, 20, 20))
        draw = ImageDraw.Draw(image)
        draw.rectangle((205, 125, 435, 143), fill=(15, 145, 35))
        draw.rectangle((220, 166, 420, 180), fill=(20, 130, 30))

        groups = _dialog_green_groups(image)

        self.assertEqual(len(groups), 2)
        self.assertAlmostEqual(groups[0], 134, delta=2)
        self.assertAlmostEqual(groups[1], 173, delta=2)
        self.assertTrue(_opponent_dialog_visible(image))

    def test_recognizes_dialog_with_no_selected_opponent(self):
        image = Image.new("RGB", (640, 510), (20, 20, 20))
        ImageDraw.Draw(image).rectangle((205, 125, 435, 143), fill=(15, 145, 35))

        self.assertEqual(len(_dialog_green_groups(image)), 1)
        self.assertTrue(_opponent_dialog_visible(image))

    def test_detects_show_your_ghost_toggle_color(self):
        image = Image.new("RGB", (640, 510), (20, 20, 20))
        draw = ImageDraw.Draw(image)
        draw.ellipse((202, 354, 214, 366), fill=(30, 170, 45))
        self.assertTrue(_show_your_ghost_enabled(image))

        draw.ellipse((202, 354, 214, 366), fill=(150, 40, 45))
        self.assertFalse(_show_your_ghost_enabled(image))

    def test_schedule_reports_background_result(self):
        configure = Mock(return_value=True)
        controller = OpponentDialogController(configure=configure)

        result = controller.schedule_for_process(123).result(timeout=1.0)

        self.assertTrue(result)
        configure.assert_called_once_with(123)

    def test_schedule_propagates_failure(self):
        controller = OpponentDialogController(
            configure=Mock(side_effect=RuntimeError("dialog missing"))
        )

        with self.assertRaisesRegex(RuntimeError, "dialog missing"):
            controller.schedule_for_process(123).result(timeout=1.0)

    @staticmethod
    def _select_opponent_dialog():
        # Green "Select Opponent" title band, as on a 1280x720 client.
        image = Image.new("RGB", (1280, 720), (20, 20, 20))
        ImageDraw.Draw(image).rectangle((433, 176, 848, 198), fill=(15, 145, 35))
        return image

    @patch("trackmania_vla.tmnf_window.time.sleep")
    @patch("trackmania_vla.tmnf_window._press_space")
    @patch("trackmania_vla.tmnf_window._window_click")
    @patch("trackmania_vla.tmnf_window._window_image")
    @patch("trackmania_vla.tmnf_window.bring_trackmania_to_front")
    @patch("trackmania_vla.tmnf_window._trackmania_window_for_pid", return_value=456)
    def test_picks_none_opponent_with_a_single_click(
        self, _find_window, _focus, window_image, window_click, press, _sleep
    ):
        gone = Image.new("RGB", (1280, 720), (0, 0, 0))
        window_image.side_effect = [self._select_opponent_dialog(), gone]

        self.assertTrue(_configure_opponent_dialog(123))

        press.assert_not_called()
        self.assertEqual(window_click.call_args_list, [unittest.mock.call(456, *NONE_OPPONENT_ROW)])

    @patch("trackmania_vla.tmnf_window.time.sleep")
    @patch("trackmania_vla.tmnf_window._press_space")
    @patch("trackmania_vla.tmnf_window._window_click")
    @patch("trackmania_vla.tmnf_window._window_image")
    @patch("trackmania_vla.tmnf_window.bring_trackmania_to_front")
    @patch("trackmania_vla.tmnf_window._trackmania_window_for_pid", return_value=456)
    def test_repeats_click_ignored_during_fade_in(
        self, _find_window, _focus, window_image, window_click, _press, _sleep
    ):
        dialog = self._select_opponent_dialog()
        gone = Image.new("RGB", (1280, 720), (0, 0, 0))
        window_image.side_effect = [dialog, dialog, gone]

        self.assertTrue(_configure_opponent_dialog(123))

        self.assertEqual(window_click.call_count, 2)

    @patch("trackmania_vla.tmnf_window.time.sleep")
    @patch("trackmania_vla.tmnf_window._press_space")
    @patch("trackmania_vla.tmnf_window._window_click")
    @patch("trackmania_vla.tmnf_window._window_image")
    @patch("trackmania_vla.tmnf_window.bring_trackmania_to_front")
    @patch("trackmania_vla.tmnf_window._trackmania_window_for_pid", return_value=456)
    def test_presses_through_load_screens_until_dialog(
        self, _find_window, _focus, window_image, window_click, press, _sleep
    ):
        loading = Image.new("RGB", (1280, 720), (0, 0, 0))
        window_image.side_effect = [loading, loading, self._select_opponent_dialog(), loading]

        self.assertTrue(_configure_opponent_dialog(123))

        # Load screens are dismissed by clicking an empty corner, never a key.
        press.assert_not_called()
        self.assertEqual(
            window_click.call_args_list,
            [unittest.mock.call(456, *LOAD_SCREEN_DISMISS_CLICK)] * 2
            + [unittest.mock.call(456, *NONE_OPPONENT_ROW)],
        )

if __name__ == "__main__":
    unittest.main()
