from typing import List
import rasterizor.cuda.voxelize as voxelize_cuda


def is_point_in_tetrahedron(
    point: List[float],
    vertex_1: List[float],
    vertex_2: List[float],
    vertex_3: List[float],
    vertex_4: List[float],
) -> bool:
    return voxelize_cuda.is_point_in_tetrahedron(
        *point, *vertex_1, *vertex_2, *vertex_3, *vertex_4
    )


def is_point_in_triangle(
    point: List[float],
    vertex_1: List[float],
    vertex_2: List[float],
    vertex_3: List[float],
) -> bool:
    return voxelize_cuda.is_point_in_triangle(
        *point, *vertex_1, *vertex_2, *vertex_3
    )


def is_projected_point_in_triangle(
    point: List[float],
    vertex_1: List[float],
    vertex_2: List[float],
    vertex_3: List[float],
) -> bool:
    return voxelize_cuda.is_projected_point_in_triangle(
        *point, *vertex_1, *vertex_2, *vertex_3
    )


def line_intersect_plane(
    line_point: List[float],
    line_direction: List[float],
    plane_p1: List[float],
    plane_p2: List[float],
    plane_p3: List[float],
):
    return voxelize_cuda.line_intersect_plane(*line_point, *line_direction, *plane_p1, *plane_p2, *plane_p3)