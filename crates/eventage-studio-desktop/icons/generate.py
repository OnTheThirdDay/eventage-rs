#!/usr/bin/env python3
"""Draw the app icon at every size the bundlers ask for.

The icon is a four-node, three-edge DAG: one root branching into two, one of
those going on. That is the shape of what the app is about — an event log whose
history forks — and it survives being 16 pixels wide, which a picture of
anything more literal does not.

Everything here is drawn from geometry rather than resampled from one large
bitmap, so the small sizes come out crisp instead of blurred. Nodes and edges
are filled by their signed distance (distance to a centre, distance to a line
segment) and the coverage becomes an alpha value, which is what antialiases the
curves and the rounded corners without an imaging library. Only zlib and struct
are used, so this runs anywhere Python does and adds no build dependency.

    python3 icons/generate.py        # rewrites every file in this directory

Committed so the icons are reproducible: to change the mark, change the
geometry and re-run, rather than editing seven binaries by hand.
"""

import math
import os
import struct
import zlib

# Studio's own palette, so the icon and the window it opens agree.
BG = (0x1E, 0x1B, 0x4B)  # indigo-950, the dark shell's ground
NODE = (0xA5, 0xB4, 0xFC)  # indigo-300
EDGE = (0x81, 0x8C, 0xF8)  # indigo-400

HERE = os.path.dirname(os.path.abspath(__file__))


def _distance_to_segment(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_squared))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def render(size):
    """The icon at `size` square, as rows of RGBA tuples."""
    scale = size / 64.0
    corner = 12 * scale
    # Laid out on a 64x64 grid, then scaled: a root, two branches, and one of
    # them continuing — deliberately asymmetric, so it reads as a history and
    # not as an ornament.
    nodes = [(32 * scale, 13 * scale), (19 * scale, 34 * scale),
             (45 * scale, 34 * scale), (45 * scale, 53 * scale)]
    edges = [(0, 1), (0, 2), (2, 3)]
    node_radius, edge_half_width = 5.2 * scale, 2.1 * scale

    rows = []
    for y in range(size):
        row = []
        for x in range(size):
            cx, cy = x + 0.5, y + 0.5

            # Rounded-square mask, as a distance to the rounded rectangle so
            # the corners get partial coverage instead of stair-stepping.
            qx = max(abs(cx - size / 2) - (size / 2 - corner), 0.0)
            qy = max(abs(cy - size / 2) - (size / 2 - corner), 0.0)
            outside = math.hypot(qx, qy) - corner
            alpha = max(0.0, min(1.0, 0.5 - outside))
            if alpha <= 0:
                row.append((0, 0, 0, 0))
                continue

            colour = BG
            nearest_edge = min(
                _distance_to_segment(cx, cy, *nodes[i], *nodes[j]) for i, j in edges
            )
            edge_coverage = max(0.0, min(1.0, (edge_half_width - nearest_edge) + 0.5))
            if edge_coverage > 0:
                colour = tuple(
                    round(c * (1 - edge_coverage) + e * edge_coverage)
                    for c, e in zip(colour, EDGE)
                )

            nearest_node = min(math.hypot(cx - nx, cy - ny) for nx, ny in nodes)
            node_coverage = max(0.0, min(1.0, (node_radius - nearest_node) + 0.5))
            if node_coverage > 0:
                colour = tuple(
                    round(c * (1 - node_coverage) + n * node_coverage)
                    for c, n in zip(colour, NODE)
                )

            row.append((*colour, round(alpha * 255)))
        rows.append(row)
    return rows


def png(rows):
    size = len(rows)
    raw = b"".join(
        b"\x00" + bytes(channel for pixel in row for channel in pixel) for row in rows
    )

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def ico(images):
    """A Windows .ico, which is a directory of embedded PNGs."""
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for size, data in images:
        # 0 means 256 in an icon directory entry, which only has a byte for it.
        entries += struct.pack(
            "<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset
        )
        blobs += data
        offset += len(data)
    return header + entries + blobs


def icns(images):
    """A macOS .icns: `icns`, a total length, then typed PNG chunks.

    The type code carries the size *and* the scale factor it is meant for, so a
    512px image appears twice — as `ic09` (512@1x) and `ic14` (256@2x) — which
    is how Retina picks the right one.
    """
    types = {
        16: [b"icp4"],
        32: [b"icp5", b"ic11"],
        64: [b"icp6", b"ic12"],
        128: [b"ic07"],
        256: [b"ic08", b"ic13"],
        512: [b"ic09", b"ic14"],
        1024: [b"ic10"],
    }
    chunks = b""
    for size, data in images:
        for code in types[size]:
            chunks += code + struct.pack(">I", len(data) + 8) + data
    return b"icns" + struct.pack(">I", len(chunks) + 8) + chunks


def main():
    # Rendered once per distinct pixel size and reused wherever it is needed.
    sizes = [16, 32, 48, 50, 64, 128, 150, 256, 512, 1024]
    encoded = {size: png(render(size)) for size in sizes}

    named = {
        "32x32.png": 32,
        "128x128.png": 128,
        "128x128@2x.png": 256,
        "icon.png": 512,
        # Windows Store packaging asks for these two by name.
        "Square150x150Logo.png": 150,
        "StoreLogo.png": 50,
    }
    for name, size in named.items():
        with open(os.path.join(HERE, name), "wb") as f:
            f.write(encoded[size])

    with open(os.path.join(HERE, "icon.ico"), "wb") as f:
        f.write(ico([(s, encoded[s]) for s in (16, 32, 48, 64, 256)]))

    with open(os.path.join(HERE, "icon.icns"), "wb") as f:
        f.write(icns([(s, encoded[s]) for s in (16, 32, 64, 128, 256, 512, 1024)]))

    for name in sorted(os.listdir(HERE)):
        if name != os.path.basename(__file__):
            path = os.path.join(HERE, name)
            print(f"  {name:24} {os.path.getsize(path):>8} bytes")


if __name__ == "__main__":
    main()
