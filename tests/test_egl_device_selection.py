import pytest

from robosuite.utils.binding_utils import _validate_mujoco_egl_device_id


def test_accepts_global_egl_index_for_single_visible_cuda_device():
    _validate_mujoco_egl_device_id("1", "5")


def test_accepts_global_egl_index_with_uuid_cuda_selection():
    _validate_mujoco_egl_device_id(
        "GPU-3971a030-93f3-8dd8-def2-ab3cc1b507f6",
        "5",
    )


def test_rejects_non_numeric_global_egl_index():
    with pytest.raises(
        AssertionError,
        match="non-negative global EGL device index",
    ):
        _validate_mujoco_egl_device_id("1", "not-a-device")
