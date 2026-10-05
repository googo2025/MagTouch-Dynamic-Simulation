"""R133-D retrained XY model UI, retaining R132 plan preflight and old models."""
from __future__ import annotations

import magtouch_acquisition_ui_v1_3_4_R132 as previous


previous.core.APP_NAME = "MagTouch Hardware Acquisition & Experiment System V1.3.4-R133"
previous.core.BUILD_ID = "20260927-V1.3.4-R133-D-COVERAGE-RETRAIN-ONLINE"
previous.core.__file__ = __file__


def main():
    return previous.main()


if __name__ == "__main__":
    main()
