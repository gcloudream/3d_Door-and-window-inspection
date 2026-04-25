from __future__ import annotations

import base64
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from point_selection.core import PointCloud, PointRecord


DEFAULT_MAX_POINTS = int(os.getenv("POINT_CLOUD_MAX_POINTS", "300000"))


@dataclass(frozen=True)
class PCDFieldSpec:
    name: str
    size: int
    type_name: str
    count: int
    byte_offset: int
    token_offset: int


def load_point_cloud(path: Path, max_points: Optional[int] = DEFAULT_MAX_POINTS) -> PointCloud:
    return load_point_cloud_from_bytes(path.name, path.read_bytes(), max_points=max_points)


def load_point_cloud_from_upload(
    filename: str,
    *,
    content: Optional[str] = None,
    content_bytes: Optional[bytes] = None,
    content_base64: Optional[str] = None,
    max_points: Optional[int] = DEFAULT_MAX_POINTS,
) -> PointCloud:
    if content_bytes is not None:
        return load_point_cloud_from_bytes(filename, content_bytes, max_points=max_points)
    if content_base64 is not None:
        return load_point_cloud_from_bytes(
            filename,
            base64.b64decode(content_base64.encode("ascii")),
            max_points=max_points,
        )
    if content is not None:
        return load_point_cloud_from_text(filename, content, max_points=max_points)
    raise ValueError("Scene upload requires file content")


def load_point_cloud_from_text(
    filename: str,
    content: str,
    max_points: Optional[int] = DEFAULT_MAX_POINTS,
) -> PointCloud:
    return load_point_cloud_from_bytes(filename, content.encode("utf-8"), max_points=max_points)


def load_point_cloud_from_bytes(
    filename: str,
    content: bytes,
    max_points: Optional[int] = DEFAULT_MAX_POINTS,
) -> PointCloud:
    suffix = Path(filename).suffix.lower()
    if suffix == ".json":
        return load_json_point_cloud(decode_text_content(content), max_points=max_points)
    if suffix == ".ply":
        return load_ascii_ply_content(decode_text_content(content), max_points=max_points)
    if suffix == ".pcd":
        return load_pcd_content(content, max_points=max_points)
    raise ValueError(f"Unsupported point cloud format: {suffix}")


def load_json_point_cloud(content: str, max_points: Optional[int] = DEFAULT_MAX_POINTS) -> PointCloud:
    payload = json.loads(content)
    raw_points = payload["points"]
    sample_step = compute_sample_step(len(raw_points), max_points)
    points = [
        PointRecord(
            point_id=entry["point_id"],
            xyz=tuple(entry["xyz"]),
            rgb=tuple(entry["rgb"]) if entry.get("rgb") is not None else None,
        )
        for entry in raw_points[::sample_step]
    ]
    return PointCloud(points)


def load_ascii_ply(path: Path, max_points: Optional[int] = DEFAULT_MAX_POINTS) -> PointCloud:
    return load_ascii_ply_content(path.read_text(), max_points=max_points)


def load_ascii_ply_content(
    content: str,
    max_points: Optional[int] = DEFAULT_MAX_POINTS,
) -> PointCloud:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError("Invalid PLY file: missing magic header")

    format_name = None
    vertex_count = None
    properties: List[str] = []
    in_vertex_element = False
    header_end_index = None

    for index, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if stripped.startswith("format "):
            format_name = stripped.split()[1]
        elif stripped.startswith("element "):
            tokens = stripped.split()
            in_vertex_element = len(tokens) >= 3 and tokens[1] == "vertex"
            if in_vertex_element:
                vertex_count = int(tokens[2])
        elif stripped.startswith("property ") and in_vertex_element:
            properties.append(stripped.split()[-1])
        elif stripped == "end_header":
            header_end_index = index
            break

    if format_name != "ascii":
        raise ValueError("Only ASCII PLY files are supported")
    if vertex_count is None:
        raise ValueError("PLY vertex element is required")
    if header_end_index is None:
        raise ValueError("PLY header is incomplete")

    required = {"x", "y", "z"}
    if not required.issubset(set(properties)):
        raise ValueError("PLY vertex properties must include x, y and z")

    property_positions: Dict[str, int] = {name: idx for idx, name in enumerate(properties)}
    vertex_lines = lines[header_end_index + 1 : header_end_index + 1 + vertex_count]
    if len(vertex_lines) != vertex_count:
        raise ValueError("PLY file ended before all vertex rows were read")

    sample_step = compute_sample_step(vertex_count, max_points)
    points: List[PointRecord] = []
    for row_index in range(0, vertex_count, sample_step):
        row = vertex_lines[row_index]
        tokens = row.split()
        if len(tokens) < len(properties):
            raise ValueError(f"PLY vertex row {row_index + 1} is missing properties")

        xyz = (
            float(tokens[property_positions["x"]]),
            float(tokens[property_positions["y"]]),
            float(tokens[property_positions["z"]]),
        )
        rgb = read_rgb(tokens, property_positions)
        points.append(PointRecord(point_id=row_index + 1, xyz=xyz, rgb=rgb))

    return PointCloud(points)


def load_pcd_content(content: bytes, max_points: Optional[int] = DEFAULT_MAX_POINTS) -> PointCloud:
    header_lines, data_offset = split_pcd_header(content)
    header = parse_pcd_header(header_lines)
    field_specs = build_pcd_field_specs(header)
    field_map = {field.name: field for field in field_specs}
    required = {"x", "y", "z"}
    if not required.issubset(field_map):
        raise ValueError("PCD fields must include x, y and z")

    data_format = header["data"]
    point_count = header["points"]
    sample_step = compute_sample_step(point_count, max_points)
    if data_format == "ascii":
        return load_ascii_pcd_points(content, data_offset, point_count, sample_step, field_map)
    if data_format == "binary":
        return load_binary_pcd_points(content, data_offset, point_count, sample_step, field_specs, field_map)
    raise ValueError(f"Unsupported PCD DATA format: {data_format}")


def load_ascii_pcd_points(
    content: bytes,
    data_offset: int,
    point_count: int,
    sample_step: int,
    field_map: Dict[str, PCDFieldSpec],
) -> PointCloud:
    data_lines = [line.strip() for line in decode_text_content(content[data_offset:]).splitlines() if line.strip()]
    if len(data_lines) < point_count:
        raise ValueError("PCD file ended before all point rows were read")

    total_token_count = sum(field.count for field in field_map.values())
    points: List[PointRecord] = []
    for row_index in range(0, point_count, sample_step):
        tokens = data_lines[row_index].split()
        if len(tokens) < total_token_count:
            raise ValueError(f"PCD row {row_index + 1} is missing values")

        xyz = (
            float(tokens[field_map["x"].token_offset]),
            float(tokens[field_map["y"].token_offset]),
            float(tokens[field_map["z"].token_offset]),
        )
        rgb = read_pcd_rgb_from_tokens(tokens, field_map)
        points.append(PointRecord(point_id=row_index + 1, xyz=xyz, rgb=rgb))

    return PointCloud(points)


def sample_point_cloud(cloud: PointCloud, max_points: Optional[int]) -> PointCloud:
    sample_step = compute_sample_step(len(cloud.points), max_points)
    return PointCloud(cloud.points[::sample_step])


def load_binary_pcd_points(
    content: bytes,
    data_offset: int,
    point_count: int,
    sample_step: int,
    field_specs: Sequence[PCDFieldSpec],
    field_map: Dict[str, PCDFieldSpec],
) -> PointCloud:
    point_stride = sum(field.size * field.count for field in field_specs)
    required_bytes = point_count * point_stride
    data = memoryview(content)[data_offset:]
    if len(data) < required_bytes:
        raise ValueError("PCD binary payload ended before all point rows were read")

    points: List[PointRecord] = []
    for row_index in range(0, point_count, sample_step):
        row_offset = row_index * point_stride
        xyz = (
            float(read_pcd_scalar_from_bytes(data, row_offset, field_map["x"])),
            float(read_pcd_scalar_from_bytes(data, row_offset, field_map["y"])),
            float(read_pcd_scalar_from_bytes(data, row_offset, field_map["z"])),
        )
        rgb = read_pcd_rgb_from_bytes(data, row_offset, field_map)
        points.append(PointRecord(point_id=row_index + 1, xyz=xyz, rgb=rgb))

    return PointCloud(points)


def split_pcd_header(content: bytes) -> tuple[list[str], int]:
    header_lines: list[str] = []
    offset = 0

    while offset < len(content):
        line_end = content.find(b"\n", offset)
        if line_end == -1:
            raise ValueError("PCD header is incomplete")

        raw_line = content[offset:line_end]
        offset = line_end + 1
        line = raw_line.decode("ascii", errors="strict").strip()
        header_lines.append(line)
        if line.upper().startswith("DATA "):
            return header_lines, offset

    raise ValueError("PCD header is incomplete")


def parse_pcd_header(header_lines: Sequence[str]) -> Dict[str, object]:
    header_values: Dict[str, List[str]] = {}
    for line in header_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        key = tokens[0].lower()
        header_values[key] = tokens[1:]

    fields = header_values.get("fields")
    sizes = header_values.get("size")
    types = header_values.get("type")
    if not fields or not sizes or not types:
        raise ValueError("PCD header must include FIELDS, SIZE and TYPE")

    counts = header_values.get("count", ["1"] * len(fields))
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("PCD field metadata lengths do not match")

    width = int(header_values.get("width", ["0"])[0])
    height = int(header_values.get("height", ["1"])[0])
    points = int(header_values.get("points", [str(width * height)])[0])
    data_values = header_values.get("data")
    if not data_values:
        raise ValueError("PCD header must include DATA")

    data_format = data_values[0].lower()
    if data_format == "binary_compressed":
        raise ValueError("Compressed PCD files are not supported")

    return {
        "fields": fields,
        "sizes": [int(value) for value in sizes],
        "types": [value.upper() for value in types],
        "counts": [int(value) for value in counts],
        "points": points,
        "data": data_format,
    }


def build_pcd_field_specs(header: Dict[str, object]) -> List[PCDFieldSpec]:
    field_specs: List[PCDFieldSpec] = []
    byte_offset = 0
    token_offset = 0
    fields = header["fields"]
    sizes = header["sizes"]
    types = header["types"]
    counts = header["counts"]

    for name, size, type_name, count in zip(fields, sizes, types, counts):
        field_specs.append(
            PCDFieldSpec(
                name=name,
                size=size,
                type_name=type_name,
                count=count,
                byte_offset=byte_offset,
                token_offset=token_offset,
            )
        )
        byte_offset += size * count
        token_offset += count
    return field_specs


def read_pcd_scalar_from_bytes(data: memoryview, row_offset: int, field: PCDFieldSpec):
    raw = bytes(data[row_offset + field.byte_offset : row_offset + field.byte_offset + field.size])
    return decode_scalar(raw, field.type_name, field.size)


def decode_scalar(raw: bytes, type_name: str, size: int):
    if type_name == "F":
        if size == 4:
            return struct.unpack("<f", raw)[0]
        if size == 8:
            return struct.unpack("<d", raw)[0]
    if type_name == "I":
        format_map = {1: "<b", 2: "<h", 4: "<i", 8: "<q"}
        if size in format_map:
            return struct.unpack(format_map[size], raw)[0]
    if type_name == "U":
        format_map = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}
        if size in format_map:
            return struct.unpack(format_map[size], raw)[0]
    raise ValueError(f"Unsupported PCD scalar type: {type_name}{size}")


def read_pcd_rgb_from_bytes(data: memoryview, row_offset: int, field_map: Dict[str, PCDFieldSpec]):
    direct_rgb = read_pcd_direct_rgb_from_bytes(data, row_offset, field_map)
    if direct_rgb is not None:
        return direct_rgb

    for name in ("rgb", "rgba"):
        field = field_map.get(name)
        if field is None:
            continue
        raw = bytes(data[row_offset + field.byte_offset : row_offset + field.byte_offset + field.size])
        return decode_packed_rgb_bytes(raw)
    return None


def read_pcd_direct_rgb_from_bytes(data: memoryview, row_offset: int, field_map: Dict[str, PCDFieldSpec]):
    names = [("red", "green", "blue"), ("r", "g", "b")]
    for red_name, green_name, blue_name in names:
        if red_name in field_map and green_name in field_map and blue_name in field_map:
            return (
                int(read_pcd_scalar_from_bytes(data, row_offset, field_map[red_name])),
                int(read_pcd_scalar_from_bytes(data, row_offset, field_map[green_name])),
                int(read_pcd_scalar_from_bytes(data, row_offset, field_map[blue_name])),
            )
    return None


def read_pcd_rgb_from_tokens(tokens: Sequence[str], field_map: Dict[str, PCDFieldSpec]):
    direct_rgb = read_pcd_direct_rgb_from_tokens(tokens, field_map)
    if direct_rgb is not None:
        return direct_rgb

    for name in ("rgb", "rgba"):
        field = field_map.get(name)
        if field is None:
            continue
        token = tokens[field.token_offset]
        return decode_packed_rgb_token(token, field.type_name, field.size)
    return None


def read_pcd_direct_rgb_from_tokens(tokens: Sequence[str], field_map: Dict[str, PCDFieldSpec]):
    names = [("red", "green", "blue"), ("r", "g", "b")]
    for red_name, green_name, blue_name in names:
        if red_name in field_map and green_name in field_map and blue_name in field_map:
            return (
                int(float(tokens[field_map[red_name].token_offset])),
                int(float(tokens[field_map[green_name].token_offset])),
                int(float(tokens[field_map[blue_name].token_offset])),
            )
    return None


def decode_packed_rgb_bytes(raw: bytes):
    if len(raw) == 4:
        packed = struct.unpack("<I", raw)[0]
    elif len(raw) == 8:
        packed = struct.unpack("<Q", raw)[0] & 0xFFFFFFFF
    else:
        raise ValueError("Packed RGB fields must be 4 or 8 bytes")
    return (
        (packed >> 16) & 0xFF,
        (packed >> 8) & 0xFF,
        packed & 0xFF,
    )


def decode_packed_rgb_token(token: str, type_name: str, size: int):
    if type_name == "F":
        if size == 4:
            raw = struct.pack("<f", float(token))
        elif size == 8:
            raw = struct.pack("<d", float(token))[:4]
        else:
            raise ValueError("Unsupported packed float RGB size")
        return decode_packed_rgb_bytes(raw)

    packed = int(float(token))
    return (
        (packed >> 16) & 0xFF,
        (packed >> 8) & 0xFF,
        packed & 0xFF,
    )


def compute_sample_step(point_count: int, max_points: Optional[int]) -> int:
    if point_count <= 0:
        return 1
    if max_points is None:
        return 1
    if max_points < 1:
        raise ValueError("max_points must be at least 1")
    return max(1, int(math.ceil(point_count / max_points)))


def decode_text_content(content: bytes) -> str:
    return content.decode("utf-8-sig")


def read_rgb(tokens: Sequence[str], property_positions: Dict[str, int]):
    names = [("red", "green", "blue"), ("r", "g", "b")]
    for red_name, green_name, blue_name in names:
        if red_name in property_positions and green_name in property_positions and blue_name in property_positions:
            return (
                int(tokens[property_positions[red_name]]),
                int(tokens[property_positions[green_name]]),
                int(tokens[property_positions[blue_name]]),
            )
    return None
