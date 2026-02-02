from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so the helpers package can be imported
# even when this script is run directly from the helpers folder.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from helpers.xml_stream_parser import (
    CoordinateDefinitionsExtractor,
    VideoDeviceExtractor,
    XmlStreamParser,
)


def main() -> None:
    # Point to the project root so the configs folder is found after moving this script.
    xml_path = (
        project_root
        / "configs"
        / "PlusDeviceSet_fCal_Epiphan_NDIPolaris_RadboudUMC_20241219_150400.xml"
    )

    parser = XmlStreamParser(str(xml_path))
    parser.register(VideoDeviceExtractor())
    parser.register(CoordinateDefinitionsExtractor())
    data = parser.parse()

    print("VideoDevice:")
    clip_rectangle_origin = data.get("VideoDevice").get("attrib").get("ClipRectangleOrigin")
    clip_rectangle_size = data.get("VideoDevice").get("attrib").get("ClipRectangleSize")

    origin_x, origin_y = map(int, clip_rectangle_origin.split())
    size_width, size_height = map(int, clip_rectangle_size.split())

if __name__ == "__main__":
    main()
