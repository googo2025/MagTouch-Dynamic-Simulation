"""R355 exact SPACE_DEPTH/SLOW prospective online test entry.

Adds only two byte-frozen UI plans to the R354 final-model application.  The
R348 inference, magnetic acquisition, full-plan execution and waypoint safety
preflight remain unchanged.  Importing the module does not connect hardware.
"""
from __future__ import annotations

import magtouch_acquisition_ui_v1_3_4_R354 as previous
from training.waypoint_xyz_protocol_r136 import FROZEN_SOURCE_PLANS


core = previous.core
core.APP_NAME = "MagTouch Hardware Acquisition & Experiment System V1.3.4-R355"
core.BUILD_ID = "20260930-V1.3.4-R355-EXACT-SPACE-DEPTH-SLOW"
core.__file__ = __file__

FROZEN_SOURCE_PLANS.update({
    "R355_SOURCE_R348_EXACT_SPACE_DEPTH_SLOW_ONLINE": (
        "R355_SOURCE_R348_EXACT_SPACE_DEPTH_SLOW_ONLINE.json",
        "65cfd0f6bf320e750daa92b2e621e795de88ffc55971658dd93a4f554ed972ab"),
    "R355_TARGET_R348_EXACT_SPACE_DEPTH_SLOW_ONLINE": (
        "R355_TARGET_R348_EXACT_SPACE_DEPTH_SLOW_ONLINE.json",
        "e536219e1450aa8df6db80992e66028e3ead39a7375d5ba9c6138b92aa1760f5"),
})


def main():
    return previous.main()


if __name__ == "__main__":
    main()
