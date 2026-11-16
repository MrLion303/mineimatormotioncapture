from __future__ import annotations

from math import acos, asin, atan2, cos, degrees, pi, sin, sqrt
from typing import Any, Iterable


MIN_VISIBILITY = 0.20
Vector = tuple[float, float, float]
Basis = tuple[Vector, Vector, Vector]
Quaternion = tuple[float, float, float, float]


def _point(pose: dict[str, Any], index: int) -> Vector | None:
    points = pose.get("world") or []
    if index >= len(points):
        return None
    value = points[index]
    if value is None or len(value) < 3:
        return None

    visibility = float(value[3]) if len(value) > 3 else 1.0
    presence = float(value[4]) if len(value) > 4 else 1.0
    if min(visibility, presence) < MIN_VISIBILITY:
        return None

    point = (float(value[0]), float(value[1]), float(value[2]))
    if not all(abs(component) < 1000 for component in point):
        return None
    return point


def _add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vector, factor: float) -> Vector:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(value: Vector) -> float:
    return sqrt(max(0.0, _dot(value, value)))


def _normalize(value: Vector) -> Vector | None:
    magnitude = _length(value)
    if magnitude < 1e-6:
        return None
    return _scale(value, 1.0 / magnitude)


def _midpoint(a: Vector, b: Vector) -> Vector:
    return _scale(_add(a, b), 0.5)


def _body_basis(pose: dict[str, Any]) -> Basis | None:
    left_shoulder = _point(pose, 11)
    right_shoulder = _point(pose, 12)
    left_hip = _point(pose, 23)
    right_hip = _point(pose, 24)
    if any(point is None for point in (left_shoulder, right_shoulder, left_hip, right_hip)):
        return None

    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)
    shoulder_width = _sub(right_shoulder, left_shoulder)
    hip_width = _sub(right_hip, left_hip)

    x_axis = _normalize(_add(shoulder_width, hip_width)) or _normalize(shoulder_width)
    if x_axis is None:
        return None

    up_raw = _sub(shoulder_mid, hip_mid)
    up_orthogonal = _sub(up_raw, _scale(x_axis, _dot(up_raw, x_axis)))
    y_axis = _normalize(up_orthogonal)
    if y_axis is None:
        return None

    z_axis = _normalize(_cross(x_axis, y_axis))
    if z_axis is None:
        return None
    return (x_axis, y_axis, z_axis)



def _root_basis(pose: dict[str, Any]) -> Basis | None:
    """Estimate pelvis facing from the hip line only.

    Knees and ankles are deliberately excluded: their cyclic motion during a
    walk must never be interpreted as a whole-character turn.
    """
    left_hip = _point(pose, 23)
    right_hip = _point(pose, 24)
    if left_hip is None or right_hip is None:
        return None

    x_axis = _normalize(_sub(right_hip, left_hip))
    if x_axis is None:
        return None

    world_up: Vector = (0.0, -1.0, 0.0)
    up_orthogonal = _sub(world_up, _scale(x_axis, _dot(world_up, x_axis)))
    y_axis = _normalize(up_orthogonal)
    if y_axis is None:
        return None
    z_axis = _normalize(_cross(x_axis, y_axis))
    if z_axis is None:
        return None
    y_axis = _normalize(_cross(z_axis, x_axis)) or y_axis
    return (x_axis, y_axis, z_axis)


def _image_point(pose: dict[str, Any], index: int) -> tuple[float, float] | None:
    points = pose.get("image") or []
    if index >= len(points):
        return None
    value = points[index]
    if value is None or len(value) < 2:
        return None
    visibility = float(value[3]) if len(value) > 3 else 1.0
    presence = float(value[4]) if len(value) > 4 else 1.0
    if min(visibility, presence) < MIN_VISIBILITY:
        return None
    return (float(value[0]), float(value[1]))


def _root_image_metrics(pose: dict[str, Any]) -> tuple[float, float, float, float, float] | None:
    """Return hip image position, metres-per-pixel and camera distance estimate."""
    image = pose.get("image") or []
    world = pose.get("world") or []
    image_size = pose.get("image_size") or (640, 480)
    if len(image_size) < 2:
        return None
    width, height = float(image_size[0]), float(image_size[1])
    if width <= 0.0 or height <= 0.0:
        return None

    left_hip = _image_point(pose, 23)
    right_hip = _image_point(pose, 24)
    if left_hip is None or right_hip is None:
        return None
    hip_x = (left_hip[0] + right_hip[0]) * 0.5
    hip_y = (left_hip[1] + right_hip[1]) * 0.5

    height_indices = (0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
    image_y: list[float] = []
    world_y: list[float] = []
    for index in height_indices:
        image_point = _image_point(pose, index)
        world_point = _point(pose, index)
        if image_point is None or world_point is None:
            continue
        image_y.append(image_point[1])
        world_y.append(world_point[1])

    if len(image_y) < 6 or len(world_y) < 6:
        return None
    pixel_height = (max(image_y) - min(image_y)) * height
    world_height = max(world_y) - min(world_y)
    if pixel_height < 12.0 or world_height < 0.25:
        return None

    metres_per_pixel = world_height / pixel_height
    focal_pixels = height / (2.0 * 0.4663076581549986)  # tan(25 degrees)
    distance = focal_pixels * metres_per_pixel
    return hip_x, hip_y, metres_per_pixel, distance, width


def _root_translation_delta(
    pose: dict[str, Any],
    reference_pose: dict[str, Any],
    units_per_metre: float,
) -> Vector | None:
    """Estimate camera-relative root translation from tracked hips."""
    current = _root_image_metrics(pose)
    reference = _root_image_metrics(reference_pose)
    if current is None or reference is None:
        return None

    current_x, current_y, current_scale, current_distance, width = current
    reference_x, reference_y, reference_scale, reference_distance, _ = reference
    image_size = pose.get("image_size") or (640, 480)
    height = float(image_size[1])
    scale = (current_scale + reference_scale) * 0.5

    delta_x = (current_x - reference_x) * width * scale
    delta_z = -(current_y - reference_y) * height * scale
    delta_y = reference_distance - current_distance
    return (
        delta_x * units_per_metre,
        delta_y * units_per_metre,
        delta_z * units_per_metre,
    )

def _to_local(vector: Vector, basis: Basis) -> Vector:
    return (
        _dot(vector, basis[0]),
        _dot(vector, basis[1]),
        _dot(vector, basis[2]),
    )


def _basis_in_parent(child: Basis, parent: Basis) -> Basis:
    return tuple(
        (
            _dot(axis, parent[0]),
            _dot(axis, parent[1]),
            _dot(axis, parent[2]),
        )
        for axis in child
    )  # type: ignore[return-value]


def _head_basis(pose: dict[str, Any]) -> Basis | None:
    nose = _point(pose, 0)
    left_side = _point(pose, 7) or _point(pose, 2)
    right_side = _point(pose, 8) or _point(pose, 5)
    if nose is None or left_side is None or right_side is None:
        return None

    x_axis = _normalize(_sub(right_side, left_side))
    if x_axis is None:
        return None
    side_mid = _midpoint(left_side, right_side)
    forward_raw = _sub(nose, side_mid)
    forward_orthogonal = _sub(forward_raw, _scale(x_axis, _dot(forward_raw, x_axis)))
    z_axis = _normalize(forward_orthogonal)
    if z_axis is None:
        return None
    y_axis = _normalize(_cross(z_axis, x_axis))
    if y_axis is None:
        return None
    z_axis = _normalize(_cross(x_axis, y_axis))
    if z_axis is None:
        return None
    return (x_axis, y_axis, z_axis)


def _quaternion_normalize(q: Quaternion) -> Quaternion:
    magnitude = sqrt(sum(value * value for value in q))
    if magnitude < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(value / magnitude for value in q)  # type: ignore[return-value]


def _quaternion_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return _quaternion_normalize((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ))


def _quaternion_conjugate(q: Quaternion) -> Quaternion:
    return (q[0], -q[1], -q[2], -q[3])


def _quaternion_from_euler_xyz(angles: tuple[float, float, float]) -> Quaternion:
    x, y, z = (value * pi / 360.0 for value in angles)
    cx, sx = cos(x), sin(x)
    cy, sy = cos(y), sin(y)
    cz, sz = cos(z), sin(z)
    return _quaternion_normalize((
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ))


def _quaternion_to_euler_xyz(
    q: Quaternion,
    reference_angles: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    """Convert to Euler angles while choosing the equivalent branch nearest the previous frame."""
    w, x, y, z = _quaternion_normalize(q)

    angle_x = atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    sin_y = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    angle_y = asin(sin_y)
    angle_z = atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    primary = tuple(degrees(value) for value in (angle_x, angle_y, angle_z))

    if reference_angles is None:
        return primary

    # XYZ Euler rotations have two equivalent representations. The principal
    # branch can jump 180 degrees in X and Z when Y crosses +/-90 degrees.
    alternate_y = 180.0 - primary[1] if primary[1] >= 0.0 else -180.0 - primary[1]
    alternate = (primary[0] + 180.0, alternate_y, primary[2] + 180.0)

    candidates = []
    for candidate in (primary, alternate):
        adjusted = tuple(
            value + 360.0 * round((reference - value) / 360.0)
            for value, reference in zip(candidate, reference_angles)
        )
        candidates.append(adjusted)
    return min(
        candidates,
        key=lambda candidate: sum(
            (value - reference) ** 2
            for value, reference in zip(candidate, reference_angles)
        ),
    )


def _quaternion_from_basis(basis: Basis) -> Quaternion:
    x_axis, y_axis, z_axis = basis
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        scale = sqrt(trace + 1.0) * 2.0
        q = (
            0.25 * scale,
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
        )
    elif m00 > m11 and m00 > m22:
        scale = sqrt(max(1e-12, 1.0 + m00 - m11 - m22)) * 2.0
        q = (
            (m21 - m12) / scale,
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
        )
    elif m11 > m22:
        scale = sqrt(max(1e-12, 1.0 + m11 - m00 - m22)) * 2.0
        q = (
            (m02 - m20) / scale,
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
        )
    else:
        scale = sqrt(max(1e-12, 1.0 + m22 - m00 - m11)) * 2.0
        q = (
            (m10 - m01) / scale,
            (m02 + m20) / scale,
            (m12 + m21) / scale,
            0.25 * scale,
        )
    return _quaternion_normalize(q)


def _basis_from_quaternion(q: Quaternion) -> Basis:
    """Build an orthonormal basis from a quaternion (basis axes are matrix columns)."""
    w, x, y, z = _quaternion_normalize(q)
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        (
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ),
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _quaternion_between_vectors(start: Vector, end: Vector) -> Quaternion:
    a = _normalize(start)
    b = _normalize(end)
    if a is None or b is None:
        return (1.0, 0.0, 0.0, 0.0)

    dot = max(-1.0, min(1.0, _dot(a, b)))
    cross = _cross(a, b)
    if dot < -0.999999:
        fallback = (1.0, 0.0, 0.0) if abs(a[0]) < 0.8 else (0.0, 1.0, 0.0)
        axis = _normalize(_cross(a, fallback)) or (0.0, 0.0, 1.0)
        return (0.0, axis[0], axis[1], axis[2])
    return _quaternion_normalize((1.0 + dot, cross[0], cross[1], cross[2]))


def _relative_basis_rotation(reference: Basis, current: Basis) -> Quaternion:
    reference_q = _quaternion_from_basis(reference)
    current_q = _quaternion_from_basis(current)
    return _quaternion_multiply(_quaternion_conjugate(reference_q), current_q)


def _base_rotation(
    base_values: dict[str, dict[str, float]] | None,
    part: str,
) -> tuple[float, float, float]:
    values = (base_values or {}).get(part, {})
    return (
        float(values.get("ROT_X", 0.0)),
        float(values.get("ROT_Y", 0.0)),
        float(values.get("ROT_Z", 0.0)),
    )


def _rotation_fields(part: str, body_bend_axes: tuple[str, ...] = ()) -> tuple[str, str, str]:
    x_field = "BEND_ANGLE_X" if part == "body" and "BEND_ANGLE_X" in body_bend_axes else "ROT_X"
    z_field = "BEND_ANGLE_Z" if part == "body" and "BEND_ANGLE_Z" in body_bend_axes else "ROT_Z"
    return x_field, "ROT_Y", z_field


def _previous_rotation_angles(
    part: str,
    base_values: dict[str, dict[str, float]] | None,
    previous_controls: dict[str, dict[str, float]] | None,
    body_bend_axes: tuple[str, ...] = (),
) -> tuple[float, float, float] | None:
    previous = (previous_controls or {}).get(part)
    if not previous:
        return None
    base = (base_values or {}).get(part, {})
    fields = _rotation_fields(part, body_bend_axes)
    return tuple(float(previous.get(field, base.get(field, 0.0))) for field in fields)  # type: ignore[return-value]


def _apply_rotation_delta(
    base_values: dict[str, dict[str, float]] | None,
    part: str,
    delta: Quaternion,
    previous_angles: tuple[float, float, float] | None = None,
    body_bend_axes: tuple[str, ...] = (),
) -> dict[str, float]:
    """Compose a parent-local motion quaternion with the rig's rest rotation."""
    base = (base_values or {}).get(part, {})
    fields = _rotation_fields(part, body_bend_axes)
    base_angles = tuple(float(base.get(field, 0.0)) for field in fields)
    rest_q = _quaternion_from_euler_xyz(base_angles)
    result_q = _quaternion_multiply(rest_q, delta)
    angles = _quaternion_to_euler_xyz(
        result_q,
        reference_angles=previous_angles or base_angles,
    )
    values = {
        "ROT_X": float(base.get("ROT_X", 0.0)),
        "ROT_Y": float(base.get("ROT_Y", 0.0)),
        "ROT_Z": float(base.get("ROT_Z", 0.0)),
    }
    for field, value in zip(fields, angles):
        values[field] = value
    return values


def _stable_perpendicular(axis: Vector) -> Vector:
    candidates: tuple[Vector, ...] = (
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    )
    reference = min(candidates, key=lambda candidate: abs(_dot(axis, candidate)))
    projected = _sub(reference, _scale(axis, _dot(reference, axis)))
    return _normalize(projected) or (0.0, 1.0, 0.0)


def _limb_basis(
    upper: Vector,
    lower: Vector,
    preferred_normal: Vector | None = None,
) -> Basis | None:
    """Build a limb orientation from its axis and bend-plane normal.

    The bend magnitude is exported separately through BEND_ANGLE_X. Using the
    bend-plane normal here captures axial twist and bend direction without
    encoding the elbow/knee bend amount a second time in ROT_X/Y/Z.
    """
    x_axis = _normalize(upper)
    if x_axis is None:
        return None

    normal = _normalize(_cross(upper, lower))
    if normal is None and preferred_normal is not None:
        normal = _normalize(
            _sub(
                preferred_normal,
                _scale(x_axis, _dot(preferred_normal, x_axis)),
            )
        )
    if normal is None:
        normal = _stable_perpendicular(x_axis)

    if preferred_normal is not None and _dot(normal, preferred_normal) < 0.0:
        normal = _scale(normal, -1.0)
    z_axis = normal
    y_axis = _normalize(_cross(z_axis, x_axis))
    if y_axis is None:
        return None
    z_axis = _normalize(_cross(x_axis, y_axis)) or z_axis
    return (x_axis, y_axis, z_axis)


def _bend_angle(start: Vector, joint: Vector, end: Vector) -> float:
    upper = _normalize(_sub(start, joint))
    lower = _normalize(_sub(end, joint))
    if upper is None or lower is None:
        return 0.0
    dot = max(-1.0, min(1.0, _dot(upper, lower)))
    interior = degrees(acos(dot))
    return max(0.0, min(150.0, 180.0 - interior))


def _limb_points(pose: dict[str, Any], indices: tuple[int, int, int]):
    points = tuple(_point(pose, index) for index in indices)
    if any(point is None for point in points):
        return None
    return points


def _limb_controls(
    pose: dict[str, Any],
    reference_pose: dict[str, Any],
    current_body: Basis,
    reference_body: Basis,
    current_root: Basis,
    reference_root: Basis,
    part: str,
    indices: tuple[int, int, int],
    base_values: dict[str, dict[str, float]] | None,
    part_parents: dict[str, str] | None = None,
    previous_controls: dict[str, dict[str, float]] | None = None,
) -> dict[str, float] | None:
    current = _limb_points(pose, indices)
    reference = _limb_points(reference_pose, indices)
    if current is None or reference is None:
        return None

    default_parent = "body" if part in ("left_arm", "right_arm") else "root"
    parent_part = (part_parents or {}).get(part, default_parent)
    if parent_part == "body":
        current_parent = current_body
        reference_parent = reference_body
    else:
        current_parent = current_root
        reference_parent = reference_root

    current_upper = _to_local(_sub(current[1], current[0]), current_parent)
    current_lower = _to_local(_sub(current[2], current[1]), current_parent)
    reference_upper = _to_local(_sub(reference[1], reference[0]), reference_parent)
    reference_lower = _to_local(_sub(reference[2], reference[1]), reference_parent)
    reference_frame = _limb_basis(reference_upper, reference_lower)
    if reference_frame is None:
        return None
    current_frame = _limb_basis(
        current_upper,
        current_lower,
        preferred_normal=reference_frame[2],
    )
    if current_frame is None:
        return None

    delta_q = _relative_basis_rotation(reference_frame, current_frame)
    result = _apply_rotation_delta(
        base_values,
        part,
        delta_q,
        previous_angles=_previous_rotation_angles(part, base_values, previous_controls),
    )

    current_bend = _bend_angle(current[0], current[1], current[2])
    reference_bend = _bend_angle(reference[0], reference[1], reference[2])
    base_bend = float((base_values or {}).get(part, {}).get("BEND_ANGLE_X", 0.0))
    result["BEND_ANGLE_X"] = max(
        0.0,
        min(180.0, base_bend + current_bend - reference_bend),
    )
    return result


def calculate_pose_controls(
    pose: dict[str, Any],
    reference_pose: dict[str, Any] | None = None,
    base_values: dict[str, dict[str, float]] | None = None,
    part_parents: dict[str, str] | None = None,
    body_bend_axes: tuple[str, ...] = (),
    previous_controls: dict[str, dict[str, float]] | None = None,
    capture_translation: bool = True,
    translation_scale: float = 18.0,
) -> dict[str, dict[str, float]]:
    """Retarget visible body motion to the selected Mine-imator rig."""
    reference_pose = reference_pose or pose
    current_root_raw = _root_basis(pose)
    reference_root_raw = _root_basis(reference_pose)
    current_body_raw = _body_basis(pose)
    reference_body_raw = _body_basis(reference_pose)

    if current_root_raw is None and current_body_raw is None:
        return {}
    if reference_root_raw is None and reference_body_raw is None:
        return {}

    current_root = current_root_raw or current_body_raw
    reference_root = reference_root_raw or reference_body_raw
    current_body = current_body_raw or current_root
    reference_body = reference_body_raw or reference_root
    if current_root is None or reference_root is None or current_body is None or reference_body is None:
        return {}

    root_delta = _relative_basis_rotation(reference_root, current_root)
    root_base = (base_values or {}).get("root", {})
    previous_root = (previous_controls or {}).get("root", {})
    previous_root_delta = (
        float(previous_root.get("ROT_Z", root_base.get("ROT_Z", 0.0)))
        - float(root_base.get("ROT_Z", 0.0))
    )
    root_angles = _quaternion_to_euler_xyz(
        root_delta,
        reference_angles=(0.0, previous_root_delta, 0.0),
    )
    root_yaw_delta = root_angles[1]

    # The supplied Mine-imator human rigs use char.ROT_Z for facing.
    root_base_angles = (
        float(root_base.get("ROT_X", 0.0)),
        float(root_base.get("ROT_Y", 0.0)),
        float(root_base.get("ROT_Z", 0.0)),
    )
    root_delta_mi = _quaternion_from_euler_xyz((0.0, 0.0, root_yaw_delta))
    root_q = _quaternion_multiply(
        _quaternion_from_euler_xyz(root_base_angles),
        root_delta_mi,
    )
    root_previous_angles = (
        tuple(
            float(previous_root.get(field, root_base.get(field, 0.0)))
            for field in ("ROT_X", "ROT_Y", "ROT_Z")
        )
        if previous_root else root_base_angles
    )
    root_output_angles = _quaternion_to_euler_xyz(
        root_q,
        reference_angles=root_previous_angles,
    )
    root_values = {
        field: float(root_base.get(field, 0.0))
        for field in ("POS_X", "POS_Y", "POS_Z")
    }
    root_values.update({
        field: value
        for field, value in zip(("ROT_X", "ROT_Y", "ROT_Z"), root_output_angles)
    })
    if capture_translation:
        translation = _root_translation_delta(
            pose,
            reference_pose,
            max(0.0, float(translation_scale)),
        )
        if translation is None:
            # When hips/feet are occluded, hold the last known root position
            # rather than snapping the character back to its original location.
            for field in ("POS_X", "POS_Y", "POS_Z"):
                root_values[field] = float(
                    previous_root.get(field, root_base.get(field, 0.0))
                )
        else:
            dx, dy, dz = translation
            root_values["POS_X"] += dx
            root_values["POS_Y"] += dy
            root_values["POS_Z"] += dz
    controls: dict[str, dict[str, float]] = {"root": root_values}

    root_yaw_quaternion = _quaternion_from_euler_xyz((0.0, root_yaw_delta, 0.0))
    body_delta = _relative_basis_rotation(reference_body, current_body)
    body_local_delta = _quaternion_multiply(
        _quaternion_conjugate(root_yaw_quaternion),
        body_delta,
    )
    body_values = _apply_rotation_delta(
        base_values,
        "body",
        body_local_delta,
        previous_angles=_previous_rotation_angles(
            "body", base_values, previous_controls, body_bend_axes
        ),
        body_bend_axes=body_bend_axes,
    )
    controls["body"] = body_values

    reference_root_q = _quaternion_from_basis(reference_root)
    current_root = _basis_from_quaternion(
        _quaternion_multiply(reference_root_q, root_yaw_quaternion)
    )

    current_head = _head_basis(pose)
    reference_head = _head_basis(reference_pose)
    if current_head is not None and reference_head is not None:
        head_parent = (part_parents or {}).get("head", "body")
        if head_parent == "body":
            current_head_local = _basis_in_parent(current_head, current_body)
            reference_head_local = _basis_in_parent(reference_head, reference_body)
        else:
            current_head_local = _basis_in_parent(current_head, current_root)
            reference_head_local = _basis_in_parent(reference_head, reference_root)
        head_delta = _relative_basis_rotation(reference_head_local, current_head_local)
        controls["head"] = _apply_rotation_delta(
            base_values,
            "head",
            head_delta,
            previous_angles=_previous_rotation_angles("head", base_values, previous_controls),
        )

    limb_specs = (
        ("left_arm", (11, 13, 15)),
        ("right_arm", (12, 14, 16)),
        ("left_leg", (23, 25, 27)),
        ("right_leg", (24, 26, 28)),
    )
    for part, indices in limb_specs:
        limb = _limb_controls(
            pose,
            reference_pose,
            current_body,
            reference_body,
            current_root,
            reference_root,
            part,
            indices,
            base_values,
            part_parents=part_parents,
            previous_controls=previous_controls,
        )
        if limb is not None:
            controls[part] = limb

    return controls


class _OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = 3.0,
        beta: float = 0.16,
        derivative_cutoff: float = 1.2,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.previous_value: float | None = None
        self.previous_derivative = 0.0
        self.previous_time: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def filter(self, value: float, timestamp: float) -> float:
        if self.previous_value is None or self.previous_time is None:
            self.previous_value = value
            self.previous_time = timestamp
            return value

        dt = max(1.0 / 120.0, min(0.25, timestamp - self.previous_time))
        derivative = (value - self.previous_value) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        filtered_derivative = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self.previous_derivative
        )
        cutoff = self.min_cutoff + self.beta * abs(filtered_derivative)
        value_alpha = self._alpha(cutoff, dt)
        filtered_value = (
            value_alpha * value
            + (1.0 - value_alpha) * self.previous_value
        )

        self.previous_value = filtered_value
        self.previous_derivative = filtered_derivative
        self.previous_time = timestamp
        return filtered_value


def stabilize_pose_samples(
    samples: Iterable[tuple[float, dict[str, Any]]],
    enabled: bool = True,
) -> list[tuple[float, dict[str, Any]]]:
    """Apply time-aware One Euro filtering to 3D landmarks before retargeting."""
    ordered = sorted(
        ((float(timestamp), pose) for timestamp, pose in samples),
        key=lambda item: item[0],
    )
    if not enabled or not ordered:
        return ordered

    filters: dict[int, tuple[_OneEuroFilter, _OneEuroFilter, _OneEuroFilter]] = {}
    image_filters: dict[int, tuple[_OneEuroFilter, _OneEuroFilter]] = {}
    stabilized: list[tuple[float, dict[str, Any]]] = []

    for timestamp, pose in ordered:
        world = pose.get("world") or []
        filtered_world = []
        for index, point in enumerate(world):
            if point is None or len(point) < 3:
                filtered_world.append(point)
                continue

            visibility = float(point[3]) if len(point) > 3 else 1.0
            presence = float(point[4]) if len(point) > 4 else 1.0
            if min(visibility, presence) < MIN_VISIBILITY:
                filtered_world.append(list(point))
                continue

            point_filters = filters.setdefault(
                index,
                (_OneEuroFilter(), _OneEuroFilter(), _OneEuroFilter()),
            )
            filtered = [
                point_filters[axis].filter(float(point[axis]), timestamp)
                for axis in range(3)
            ]
            filtered.extend(float(value) for value in point[3:])
            filtered_world.append(filtered)

        filtered_image = []
        for index, point in enumerate(pose.get("image") or []):
            if point is None or len(point) < 2:
                filtered_image.append(point)
                continue
            visibility = float(point[3]) if len(point) > 3 else 1.0
            presence = float(point[4]) if len(point) > 4 else 1.0
            if min(visibility, presence) < MIN_VISIBILITY:
                filtered_image.append(list(point))
                continue
            point_filters = image_filters.setdefault(
                index,
                (_OneEuroFilter(), _OneEuroFilter()),
            )
            filtered = [
                point_filters[axis].filter(float(point[axis]), timestamp)
                for axis in range(2)
            ]
            filtered.extend(float(value) for value in point[2:])
            filtered_image.append(filtered)

        filtered_pose = dict(pose)
        filtered_pose["world"] = filtered_world
        if "image" in pose:
            filtered_pose["image"] = filtered_image
        stabilized.append((timestamp, filtered_pose))

    return stabilized
