from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol
import xml.etree.ElementTree as ET


def _local_name(tag: str) -> str:
    """
    Convert a namespaced XML tag into a plain tag name.

    ElementTree represents namespaced tags like:
        "{namespace-uri}TagName"

    Most of our parsing logic only cares about "TagName", so we strip the
    "{...}" prefix when present.
    """
    if tag.startswith("{"):
        # Example: "{http://example}Device" -> "Device"
        return tag.split("}", 1)[1]
    # No namespace prefix -> already a simple tag name.
    return tag


def _clean_attrib(attrib: Dict[str, str]) -> Dict[str, str]:
    """
    Return a copy of the element attributes with cleaned (non-namespaced) keys.

    Some XML files include namespaced attribute names too. Cleaning them here
    ensures downstream code can reliably use simple keys like "Id" or "Matrix".
    """
    return {_local_name(k): v for k, v in attrib.items()}


def _normalize_text(text: Optional[str]) -> Optional[str]:
    """
    Normalize element text so consumers don't have to handle whitespace noise.

    - `None` stays `None` (meaning: there was no text node)
    - whitespace-only strings become `None` (meaning: no meaningful content)
    - otherwise we return the trimmed string
    """
    if text is None:
        return None
    # Remove leading/trailing whitespace that often appears due to formatting/indentation.
    stripped = text.strip()
    # Treat empty-after-strip as "no useful text".
    return stripped if stripped else None


def _parse_matrix(value: str) -> List[List[float]]:
    """
    Parse a matrix encoded as a string into a list-of-lists of floats.

    Input in XML is commonly a string like:
        "1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1"
    or it may contain commas:
        "1,0,0,0,0,1,0,0,..."

    We:
    - split the string into tokens
    - convert tokens that look like numbers into floats (ignore any bad tokens)
    - if there are 4*N numbers, we group them into rows of 4 (common transform matrix layout)
    - otherwise we return a single row with all numbers (so we don't silently lose data)
    """
    # Convert commas into spaces so splitting works whether commas or spaces are used.
    tokens = value.replace(",", " ").split()
    numbers: List[float] = []
    # Convert tokens into floats; skip tokens that are not numeric.
    for token in tokens:
        try:
            numbers.append(float(token))
        except ValueError:
            # Some feeds include extra separators or formatting strings; ignore them.
            continue
    # No numeric values -> represent "no matrix" as an empty list.
    if not numbers:
        return []
    # If we have 4, 8, 12, 16, ... values, interpret them as rows of 4.
    if len(numbers) % 4 == 0:
        return [numbers[i:i + 4] for i in range(0, len(numbers), 4)]
    # Unexpected count -> keep everything in one row so the caller can decide what to do.
    return [numbers]


class Extractor(Protocol):
    """
    Interface for "extractor" plugins used by `XmlStreamParser`.

    The parser is responsible for streaming through the XML and firing events.
    An extractor is responsible for listening to those events and building a
    result specific to one part of the document.
    """
    name: str

    def start(self, tag: str, elem: ET.Element) -> None:
        ...

    def end(self, tag: str, elem: ET.Element) -> None:
        ...

    def result(self) -> Any:
        ...


class BaseExtractor:
    """
    Convenience base class for extractors.

    Most extractors only need to override a subset of the methods. This base
    class provides safe defaults that do nothing.
    """
    name = "Base"

    def start(self, tag: str, elem: ET.Element) -> None:
        # Called when an element start tag is seen; default is "do nothing".
        return None

    def end(self, tag: str, elem: ET.Element) -> None:
        # Called when an element end tag is seen; default is "do nothing".
        return None

    def result(self) -> Any:
        # Called after parsing finishes; default is no collected data.
        return None


class XmlStreamParser:
    """
    Stream (event-based) XML parser that supports pluggable "extractors".

    Why stream?
    - Some XML files can be large; loading the whole document into memory is slow
      and may run out of RAM.

    How it works:
    - We use ElementTree.iterparse() which yields ("start"/"end", element) events.
    - For each event we notify every registered extractor.
    - After "end" events we aggressively clear and detach nodes to free memory.
    """
    def __init__(self, xml_path: str) -> None:
        # Store where the XML file lives on disk.
        self._xml_path = xml_path
        # Extractors are small objects that listen to XML events and collect data.
        self._extractors: List[Extractor] = []

    def register(self, extractor: Extractor) -> None:
        """
        Register an extractor that will receive XML start/end events.

        You can register multiple extractors to pull different pieces of data
        from the same XML stream in one pass.
        """
        self._extractors.append(extractor)

    def parse(self) -> Dict[str, Any]:
        """
        Parse the XML file and return a dict mapping extractor name -> extractor result.

        Each extractor is responsible for keeping its own state. This method
        just coordinates the event flow and memory cleanup.
        """
        # Create an iterator that produces streaming events.
        context = ET.iterparse(self._xml_path, events=("start", "end"))
        # Keep a stack of ElementTree nodes so we can detach children after we're done with them.
        stack: List[ET.Element] = []

        for event, elem in context:
            # Normalize the tag for consistent comparisons (ignore namespaces).
            tag = _local_name(elem.tag)

            if event == "start":
                # Push element onto the stack so we know its parent when it ends.
                stack.append(elem)
                # Tell extractors we have entered a new element.
                for extractor in self._extractors:
                    extractor.start(tag, elem)
            else:
                # Tell extractors the element is complete (text/attributes are stable now).
                for extractor in self._extractors:
                    extractor.end(tag, elem)

                # --- Memory cleanup ---
                # ElementTree keeps references between parents and children. If we don't clear
                # and detach, a long document can keep growing in memory as we parse it.
                elem.clear()
                if stack:
                    # Pop the current element (the one that just ended).
                    current = stack.pop()
                    if stack:
                        try:
                            # Remove the child from its parent so the parent does not keep it alive.
                            stack[-1].remove(current)
                        except ValueError:
                            # In some edge cases the child may already be detached; safe to ignore.
                            pass

        # Ask each extractor for its final collected result.
        return {extractor.name: extractor.result() for extractor in self._extractors}


class VideoDeviceExtractor(BaseExtractor):
    name = "VideoDevice"

    """
    Extract the XML subtree for a specific <Device> element.

    Goal:
    - Find a <Device Id="..."> element matching `target_device_id`
    - Build a lightweight Python dict tree for that device (tag/attributes/text/children)
    - Ignore everything else in the XML to keep processing fast
    """
    def __init__(self, target_device_id: str = "VideoDevice") -> None:
        # Which <Device Id="..."> should we capture?
        self._target_device_id = target_device_id
        # Whether we are currently inside the matching <Device> block.
        self._active = False
        # How deep we are within the captured device (used to know when we exit it).
        self._depth = 0
        # Stack of dict nodes representing the captured subtree being built.
        self._stack: List[Dict[str, Any]] = []
        # Root dict for the captured device (returned by result()).
        self._root: Optional[Dict[str, Any]] = None

    def start(self, tag: str, elem: ET.Element) -> None:
        """
        Handle a start tag.

        If we are not active yet, we look for the matching <Device>.
        If we are active, we add a new child node to the current node.
        """
        if not self._active:
            # Wait until we reach the <Device> element with the desired Id.
            if tag == "Device" and elem.attrib.get("Id") == self._target_device_id:
                # We found it: start capturing this subtree.
                self._active = True
                self._depth = 1
                # Create the root node representation.
                node = {
                    "tag": tag,
                    "attrib": _clean_attrib(elem.attrib),
                    "text": None,
                    "children": [],
                }
                self._root = node
                self._stack = [node]
            return

        # Inside the captured device: every new element becomes a new node in our dict tree.
        self._depth += 1
        node = {
            "tag": tag,
            "attrib": _clean_attrib(elem.attrib),
            "text": None,
            "children": [],
        }
        # Attach the new node to the current top-of-stack node as a child.
        self._stack[-1]["children"].append(node)
        # Push the new node so any nested elements become its children.
        self._stack.append(node)

    def end(self, tag: str, elem: ET.Element) -> None:
        """
        Handle an end tag.

        When an element ends we can safely read its text and finalize its node.
        We also update depth; if depth reaches 0 we just exited the target device.
        """
        if not self._active:
            return

        # Pop the node we are finishing (it should correspond to this end event).
        node = self._stack.pop()
        # Store normalized text so consumers don't have to strip whitespace themselves.
        node["text"] = _normalize_text(elem.text)

        # Decrease depth; when it hits 0 we have closed the captured <Device>.
        self._depth -= 1
        if self._depth == 0:
            # Stop capturing; any further elements are outside the target device.
            self._active = False

    def result(self) -> Optional[Dict[str, Any]]:
        """
        Return the captured <Device> subtree as a nested dict structure.

        Returns None if the requested device was not found in the XML.
        """
        return self._root


class CoordinateDefinitionsExtractor(BaseExtractor):
    name = "CoordinateDefinitions"

    """
    Extract coordinate transforms under <CoordinateDefinitions>.

    Goal:
    - Watch for the <CoordinateDefinitions> section
    - For every <Transform ...> element inside it, capture key attributes and parse the Matrix
    - Return a list of transforms in a simple JSON-like structure
    """
    def __init__(self) -> None:
        # Start with a clean slate before parsing begins.
        self._active = False
        self._depth = 0
        self._transforms: List[Dict[str, Any]] = []

    def start(self, tag: str, elem: ET.Element) -> None:
        """
        Handle a start tag.

        We toggle "active" on entering <CoordinateDefinitions> and then count depth
        so we know when we have fully exited that section.
        """
        if not self._active:
            if tag == "CoordinateDefinitions":
                # Entered the block we care about.
                self._active = True
                self._depth = 1
            return
        # We're inside CoordinateDefinitions, so nested elements increase depth.
        self._depth += 1

    def end(self, tag: str, elem: ET.Element) -> None:
        """
        Handle an end tag.

        We only capture data while inside CoordinateDefinitions. We capture each
        Transform when it ends, then decrease depth and deactivate when we exit.
        """
        if not self._active:
            return

        if tag == "Transform":
            # Collect relevant attributes. Any extra attributes are preserved under "attrib".
            attrib = _clean_attrib(elem.attrib)
            matrix_raw = attrib.get("Matrix", "")
            transform = {
                "from": attrib.get("From"),
                "to": attrib.get("To"),
                "matrix": _parse_matrix(matrix_raw),
                "error": attrib.get("Error"),
                "date": attrib.get("Date"),
                "attrib": {
                    k: v
                    for k, v in attrib.items()
                    if k not in {"From", "To", "Matrix", "Error", "Date"}
                },
            }
            # Store each transform so callers can iterate through them later.
            self._transforms.append(transform)

        # Every end event inside the block decreases depth.
        self._depth -= 1
        if self._depth == 0:
            # We have closed the original <CoordinateDefinitions> element.
            self._active = False

    def result(self) -> Dict[str, Any]:
        """
        Return the collected transforms.

        The output is shaped like {"transforms": [...]} to be easy to merge into
        the parser's overall dict-of-results output.
        """
        return {"transforms": self._transforms}
