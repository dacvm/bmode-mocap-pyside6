import asyncio
import xml.etree.ElementTree as ElementTree

import qtm_rt

QTM_HOST = "127.0.0.1"
QTM_PORT = 22223
QTM_PROTOCOL_VERSION = "1.20"

# Keep body names global so the packet callback stays simple for juniors.
BODY_NAMES = []


# Summary:
# - Parses QTM 6D parameters XML and returns a list of rigid body names.
# - Input: xml_string (str).
# - Returns: list of rigid body names (list[str]).
def _parse_6d_body_names(xml_string):
    # Return early if there is no XML to parse.
    if not xml_string:
        return []

    # Parse the XML so we can extract body names from the 6D section.
    try:
        root = ElementTree.fromstring(xml_string)
    except ElementTree.ParseError:
        # Avoid crashing the stream if XML is malformed.
        print("Failed to parse 6D parameters XML.", flush=True)
        return []

    # Find the 6D block and collect all <Body><Name> entries.
    the_6d = root.find(".//The_6D")
    if the_6d is None:
        return []

    # Build a list of body names in the same order as QTM reports them.
    body_names = []
    for body in the_6d.findall("Body"):
        name_element = body.find("Name")
        if name_element is not None and name_element.text:
            body_names.append(name_element.text.strip())

    return body_names


# Summary:
# - Requests 6D parameters from QTM to map body index -> body name.
# - Input: connection (QTMConnection).
# - Returns: list of rigid body names (list[str]).
async def _load_body_names(connection):
    # Ask QTM for the 6D parameters XML so we can map names by index.
    xml_string = await connection.get_parameters(parameters=["6d"])

    # Parse the XML into a simple name list for the packet callback.
    body_names = _parse_6d_body_names(xml_string)

    # Print a helpful status so users know if names were found.
    if body_names:
        print(f"Loaded {len(body_names)} body name(s) from QTM.", flush=True)
    else:
        print("No body names found in QTM 6D parameters.", flush=True)

    return body_names


# Summary:
# - Prints 6D residual data for each incoming QTM frame.
# - Input: packet (QRTPacket) from the QTM real-time stream.
# - Returns: None.
def _on_packet(packet):

    # Extract 6D residual data from the packet so we can print rigid bodies.
    result = packet.get_6d_residual()
    if result is None:
        # Skip frames that do not include 6D residual data.
        print(f"Frame {packet.framenumber} | No 6D residual data", flush=True)
        return

    # Unpack the header and rigid body data once we know it exists.
    header, bodies = result

    # Show a compact header so we know each frame arrived.
    print(f"Frame {packet.framenumber} | {header}", flush=True)

    # Print each rigid body with 6D values rounded to 2 decimals.
    for index, body in enumerate(bodies):
        # Use the known name list or fall back to a generic label.
        label = BODY_NAMES[index] if index < len(BODY_NAMES) else f"Body{index + 1}"

        # Unpack the body tuple:
        # - position, already a tuple of 3, or access it by [0-2]
        # - rotation, access the .matrix field from structure, or by [0]
        # - residual, access it by .residual field from the structure, or by [0]
        position, rotation, residual = body
        x, y, z = position
        r11, r12, r13, r21, r22, r23, r31, r32, r33 = rotation.matrix
        res = residual.residual

        # Print the label with position, rotation matrix, and residual.
        print(
            f"\t{label}: "
            f"X={x:.2f}, Y={y:.2f}, Z={z:.2f} | "
            f"R=[{r11:.2f} {r12:.2f} {r13:.2f}; "
            f"{r21:.2f} {r22:.2f} {r23:.2f}; "
            f"{r31:.2f} {r32:.2f} {r33:.2f}] | "
            f"Residual={res:.2f}",
            flush=True,
        )


# Summary:
# - Connects to QTM and starts streaming 6D residual packets.
# - Input: None.
# - Returns: None.
async def _stream_markers():
    # Connect to the local QTM instance using the known protocol version.
    connection = await qtm_rt.connect(
        QTM_HOST,
        port=QTM_PORT,
        version=QTM_PROTOCOL_VERSION,
    )

    # Stop early if the connection did not succeed.
    if connection is None:
        print(f"Could not connect to QTM at {QTM_HOST}:{QTM_PORT}", flush=True)
        return

    # Load rigid body names once so the callback can label each body.
    global BODY_NAMES
    BODY_NAMES = await _load_body_names(connection)

    # Subscribe to 6D residual frames and route packets to the callback.
    await connection.stream_frames(components=["6dres"], on_packet=_on_packet)

    # Keep the coroutine alive so streaming continues until the user stops it.
    await asyncio.Event().wait()


# Summary:
# - Runs the asyncio loop so this script works from PyCharm.
# - Input: None.
# - Returns: None.
def _main():
    # Use asyncio.run for a clean, simple event loop setup.
    try:
        asyncio.run(_stream_markers())
    except KeyboardInterrupt:
        # Allow a clean stop when the user presses the stop button in PyCharm.
        print("Streaming stopped by user.", flush=True)


if __name__ == "__main__":
    _main()
