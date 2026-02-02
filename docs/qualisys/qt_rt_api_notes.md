# `qtm_rt` API Notes (Qualisys QTM Real-time SDK for Python)

This file is meant to be a **ground-truth reference** for code generation.  
**Rule:** only use APIs and names documented here; if something isn’t listed, don’t invent it—either inspect the installed package locally or consult the RT protocol docs.

---

## What this SDK is

- `qtm-rt` is the official Qualisys Python SDK that implements the **QTM Real-Time Server (RT) protocol**. :contentReference[oaicite:0]{index=0}
- Install name vs import name:
  - Install: `qtm-rt` (pip)
  - Import: `qtm_rt` (Python module)
- Package rename history:
  - The SDK was renamed to `qtm_rt` (`qtm-rt` on pip); older versions are available as `qtm`. :contentReference[oaicite:1]{index=1}

---

## QTM prerequisites & operating modes (what must be true in QTM)

### Real-time server & ports
- QTM streams data over **TCP/IP** (and can stream over **UDP/IP** depending on protocol mode/config) for real-time output. :contentReference[oaicite:2]{index=2}
- QTM’s RT protocol documentation is included with the QTM installer and is also available online. :contentReference[oaicite:3]{index=3}

### Streaming live vs streaming from file
- Real-time processing is active in **preview mode**. :contentReference[oaicite:4]{index=4}
- You can stream from a recorded file by using **“Play with Real-Time output”** (Play menu; Ctrl+Shift+Space). :contentReference[oaicite:5]{index=5}
- Images are only supported in preview mode (not when streaming from file). :contentReference[oaicite:6]{index=6}

### Data availability & buffering realities
- Analog and force data can be buffered; **not every marker frame will contain new analog/force samples**. :contentReference[oaicite:7]{index=7}
- When there is no ongoing measurement, the RT server may send an **empty frame**. :contentReference[oaicite:8]{index=8}

---

## RT protocol basics you need for correct client behavior

### Polling vs streaming
QTM has two general patterns for getting frames: :contentReference[oaicite:9]{index=9}

1. **Polling**: request a single frame when you want it (`GetCurrentFrame` conceptually; exposed in Python as `get_current_frame()`).
2. **Streaming**: tell QTM to continuously push frames (`StreamFrames` conceptually; exposed in Python as `stream_frames()`).

### Streaming rate control (important to prevent lag)
When streaming frames, the effective send rate is constrained by:
- measurement frequency,
- real-time processing frequency,
- processing time,
- the client-requested frequency. :contentReference[oaicite:10]{index=10}

If you request a rate your client can’t keep up with, you can get buffering and lag (especially over TCP). :contentReference[oaicite:11]{index=11}

---

## Core entry point

### `qtm_rt.connect(...) -> QRTConnection | None`
Signature (async):  
`async qtm_rt.connect(host, port=22223, version='1.25', on_event=None, on_disconnect=None, timeout=5, loop=None) -> QRTConnection` :contentReference[oaicite:12]{index=12}

Key notes:
- `host`: IP/hostname of the machine running QTM.
- `port`: should be the QTM port configured for **little endian** communication. :contentReference[oaicite:13]{index=13}
- `version`: RT protocol version; the SDK does not support versions older than **1.8**. :contentReference[oaicite:14]{index=14}
- Returns `None` if connection fails (common pattern in examples). :contentReference[oaicite:15]{index=15}
- `on_event`: callback for QTM events.
- `on_disconnect`: callback invoked if disconnected.

---

## `QRTConnection` (returned by `connect()`)

`class qtm_rt.QRTConnection(protocol: QTMProtocol, timeout)` :contentReference[oaicite:16]{index=16}

### Connection / status
- `disconnect()` — disconnect from QTM. :contentReference[oaicite:17]{index=17}
- `has_transport()` — check if connected. :contentReference[oaicite:18]{index=18}
- `async qtm_version()` — get QTM version. :contentReference[oaicite:19]{index=19}
- `async byte_order()` — byte order in use (expected little endian). :contentReference[oaicite:20]{index=20}

### Events
- `async get_state() -> QRTEvent` — get latest state change; triggers `on_event` if set. :contentReference[oaicite:21]{index=21}
- `async await_event(event=None, timeout=30) -> QRTEvent` — wait for any event or a specific one. :contentReference[oaicite:22]{index=22}

### Parameters & configuration
- `async get_parameters(parameters=None) -> str`  
  Returns **XML** settings for requested components.  
  `parameters` can be `'all'` or a list including:  
  `'general'`, `'3d'`, `'6d'`, `'analog'`, `'force'`, `'gazevector'`, `'eyetracker'`, `'image'`, `'skeleton'`, `'skeleton:global'`, `'calibration'`. :contentReference[oaicite:23]{index=23}

- `async send_xml(xml: str)`  
  Used to update QTM settings with an XML document. :contentReference[oaicite:24]{index=24}

### Frame access (polling)
- `async get_current_frame(components=None) -> QRTPacket`  
  `components` can include any combination of:  
  `'2d'`, `'2dlin'`, `'3d'`, `'3dres'`, `'3dnolabels'`, `'3dnolabelsres'`,  
  `'analog'`, `'analogsingle'`, `'force'`, `'forcesingle'`,  
  `'6d'`, `'6dres'`, `'6deuler'`, `'6deulerres'`,  
  `'gazevector'`, `'eyetracker'`, `'image'`, `'timecode'`,  
  `'skeleton'`, `'skeleton:global'`. :contentReference[oaicite:25]{index=25}

### Frame access (streaming)
- `async stream_frames(frames='allframes', components=None, on_packet=None) -> str`  
  Streams frames until `stream_frames_stop()` is called. :contentReference[oaicite:26]{index=26}  
  - `frames`:
    - `'allframes'`
    - `'frequency:n'`
    - `'frequencydivisor:n'` :contentReference[oaicite:27]{index=27}
  - `components`: same component list as `get_current_frame()` (see above). :contentReference[oaicite:28]{index=28}
  - Returns `'Ok'` if successful. :contentReference[oaicite:29]{index=29}

- `async stream_frames_stop()` — stop streaming frames. :contentReference[oaicite:30]{index=30}

### Controlling QTM (capture / file / projects)
Many control operations require taking control first:

- `async take_control(password)` — take control; password is set in QTM. :contentReference[oaicite:31]{index=31}
- `async release_control()` — release control. :contentReference[oaicite:32]{index=32}
- `async new()` — create a new measurement. :contentReference[oaicite:33]{index=33}
- `async close()` — close a measurement. :contentReference[oaicite:34]{index=34}
- `async load(filename)` — load a measurement. :contentReference[oaicite:35]{index=35}
- `async save(filename, overwrite=False)` — save a measurement. :contentReference[oaicite:36]{index=36}
- `async load_project(project_path)` — load a project. :contentReference[oaicite:37]{index=37}
- `async start(rtfromfile=False)` — start RT from file; requires control. :contentReference[oaicite:38]{index=38}
- `async stop()` — stop RT from file. :contentReference[oaicite:39]{index=39}
- `async trig()` — software/wireless trigger (only if QTM configured accordingly). :contentReference[oaicite:40]{index=40}
- `async set_qtm_event(event=None)` — set an event in QTM. :contentReference[oaicite:41]{index=41}
- `async calibrate(timeout=600) -> str` — start calibration and return calibration result XML. :contentReference[oaicite:42]{index=42}

---

## `QRTPacket` (the data object you receive)

`class qtm_rt.QRTPacket(data)` :contentReference[oaicite:43]{index=43}

### Component presence checks (important!)
- Packets can contain different components across frames (especially when mixing camera data with analog/force). :contentReference[oaicite:44]{index=44}
- Use the component enum to check presence:
  - `from qtm_rt.packet import QRTComponentType`
  - `if QRTComponentType.Component3d in packet.components: ...` :contentReference[oaicite:45]{index=45}
- Component retriever functions return `None` if the component is not in the packet. :contentReference[oaicite:46]{index=46}

### Common per-packet fields
- Example usage shows `packet.framenumber`. :contentReference[oaicite:47]{index=47}

### Data getters (returns are “(header/info, data)” in typical patterns)
Packet getters documented on the SDK docs page include: :contentReference[oaicite:48]{index=48}
- `get_analog(...)`
- `get_analog_single(...)`
- `get_force(...)`
- `get_force_single(...)`
- `get_6d(...)`
- `get_6d_residual(...)`
- `get_6d_euler(...)`
- `get_6d_euler_residual(...)`
- `get_image(...)`
- `get_3d_markers(...)`
- `get_3d_markers_residual(...)`
- `get_3d_markers_no_label(...)`
- `get_3d_markers_no_label_residual(...)`
- `get_2d_markers(..., index=None)` — `index` selects camera. :contentReference[oaicite:49]{index=49}
- `get_2d_markers_linearized(..., index=None)` — `index` selects camera. :contentReference[oaicite:50]{index=50}
- `get_skeletons(...)`
- `get_gaze_vectors(...)`
- `get_eye_trackers(...)`

---

## Events: `QRTEvent`
`class qtm_rt.QRTEvent(...)` enumerates QTM event types. Examples include: :contentReference[oaicite:51]{index=51}
- `EventConnected`
- `EventConnectionClosed`
- `EventCaptureStarted`
- `EventCaptureStopped`
- `EventCalibrationStarted`
- `EventCalibrationStopped`
- `EventRTfromFileStarted`
- `EventRTfromFileStopped`
- `EventTrigger`
- (and others listed in the enum)

Use:
- `on_event` callback in `connect()`, or
- `await_event(...)` to wait for an event. :contentReference[oaicite:52]{index=52}

---

## Component types: `QRTComponentType`
`class qtm_rt.packet.QRTComponentType(...)` lists component constants like: :contentReference[oaicite:53]{index=53}
- `Component3d`
- `Component3dNoLabels`
- `ComponentAnalog`
- `ComponentForce`
- `Component6d`
- `Component6dEuler`
- `Component2d`
- `Component2dLin`
- `Component3dRes`
- `Component3dNoLabelsRes`
- `Component6dRes`
- (and others continuing in the enum)

---

## Minimal patterns (copy/paste starting points)

### Minimal: connect and stream 3D markers forever
```python
import asyncio
import qtm_rt

def on_packet(packet):
    # Always assume components can be missing in a given packet.
    # Use getters; they return None if absent.
    header_markers = packet.get_3d_markers()
    if header_markers is None:
        return
    header, markers = header_markers
    print("Frame:", packet.framenumber, "Markers:", len(markers))

async def main():
    conn = await qtm_rt.connect("127.0.0.1")
    if conn is None:
        raise RuntimeError("Could not connect to QTM")
    await conn.stream_frames(components=["3d"], on_packet=on_packet)

if __name__ == "__main__":
    asyncio.get_event_loop().create_task(main())
    asyncio.get_event_loop().run_forever()
