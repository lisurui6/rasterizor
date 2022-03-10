#include <torch/torch.h>

#include <vector>

// CUDA forward declarations
bool is_point_in_tetrahedron_cuda(
    float point[3],
    float vertex_1[3],
    float vertex_2[3],
    float vertex_3[3],
    float vertex_4[3]
);

bool is_point_in_triangle_cuda(
    float point[2],
    float vertex_1[2],
    float vertex_2[2],
    float vertex_3[2]
);

bool is_projected_point_in_triangle_cuda(
    float point[3],
    float vertex_1[3],
    float vertex_2[3],
    float vertex_3[3]
);


std::vector<float> line_intersect_plane_cuda(
    float line_point[3],
    float line_direction[3],
    float plane_p1[3],
    float plane_p2[3],
    float plane_p3[3]
);


at::Tensor forward_facet_index_map_cuda(
        at::Tensor facets,
        at::Tensor facet_index_map,
        int voxel_width,
        int voxel_depth,
        int voxel_height
    );

at::Tensor backward_voxel_map_cuda(
        at::Tensor facets,
        at::Tensor facet_index_map,
        at::Tensor alpha_map,
        at::Tensor grad_alpha_map,
        at::Tensor grad_facets,
        int voxel_width,
        int voxel_depth,
        int voxel_height,
        float eps,
        float eps_in);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor forward_facet_index_map(
        at::Tensor facets,
        at::Tensor facet_index_map,
        int voxel_width,
        int voxel_depth,
        int voxel_height
    ) {

    CHECK_INPUT(facets);
    CHECK_INPUT(facet_index_map);

    return forward_facet_index_map_cuda(facets, facet_index_map,
                                       voxel_width, voxel_depth, voxel_height);
}


at::Tensor backward_voxel_map(
        at::Tensor facets,
        at::Tensor facet_index_map,
        at::Tensor alpha_map,
        at::Tensor grad_alpha_map,
        at::Tensor grad_facets,
        int voxel_width,
        int voxel_depth,
        int voxel_height,
        float eps,
        float eps_in) {

    CHECK_INPUT(facets);
    CHECK_INPUT(facet_index_map);
    CHECK_INPUT(alpha_map);
    CHECK_INPUT(grad_alpha_map);
    CHECK_INPUT(grad_facets);

    return backward_voxel_map_cuda(facets, facet_index_map, alpha_map,
                                   grad_alpha_map, grad_facets,
                                   voxel_width, voxel_depth, voxel_height, eps, eps_in);
}

bool is_point_in_tetrahedron(
    float point_x,
    float point_y,
    float point_z,
    float vertex_1_x,
    float vertex_1_y,
    float vertex_1_z,
    float vertex_2_x,
    float vertex_2_y,
    float vertex_2_z,
    float vertex_3_x,
    float vertex_3_y,
    float vertex_3_z,
    float vertex_4_x,
    float vertex_4_y,
    float vertex_4_z
) {
    float point[3] = { point_x, point_y, point_z };
    float vertex_1[3] = { vertex_1_x, vertex_1_y, vertex_1_z };
    float vertex_2[3] = { vertex_2_x, vertex_2_y, vertex_2_z };
    float vertex_3[3] = { vertex_3_x, vertex_3_y, vertex_3_z };
    float vertex_4[3] = { vertex_4_x, vertex_4_y, vertex_4_z };

    return is_point_in_tetrahedron_cuda(point, vertex_1, vertex_2, vertex_3, vertex_4);
}


bool is_point_in_triangle(
    float point_x,
    float point_y,
    float vertex_1_x,
    float vertex_1_y,
    float vertex_2_x,
    float vertex_2_y,
    float vertex_3_x,
    float vertex_3_y
) {
    float point[2] = { point_x, point_y };
    float vertex_1[2] = { vertex_1_x, vertex_1_y };
    float vertex_2[2] = { vertex_2_x, vertex_2_y };
    float vertex_3[2] = { vertex_3_x, vertex_3_y };

    return is_point_in_triangle_cuda(point, vertex_1, vertex_2, vertex_3);
}


bool is_projected_point_in_triangle(
    float point_x,
    float point_y,
    float point_z,
    float vertex_1_x,
    float vertex_1_y,
    float vertex_1_z,
    float vertex_2_x,
    float vertex_2_y,
    float vertex_2_z,
    float vertex_3_x,
    float vertex_3_y,
    float vertex_3_z
) {
    float point[3] = { point_x, point_y, point_z };
    float vertex_1[3] = { vertex_1_x, vertex_1_y, vertex_1_z };
    float vertex_2[3] = { vertex_2_x, vertex_2_y, vertex_2_z };
    float vertex_3[3] = { vertex_3_x, vertex_3_y, vertex_3_z };

    return is_projected_point_in_triangle_cuda(point, vertex_1, vertex_2, vertex_3);
}



std::vector<float> line_intersect_plane(
    float line_point_x,
    float line_point_y,
    float line_point_z,
    float line_direction_x,
    float line_direction_y,
    float line_direction_z,
    float plane_p1_x,
    float plane_p1_y,
    float plane_p1_z,
    float plane_p2_x,
    float plane_p2_y,
    float plane_p2_z,
    float plane_p3_x,
    float plane_p3_y,
    float plane_p3_z
) {
    float line_point[3] = { line_point_x, line_point_y, line_point_z };
    float line_direction[3] = { line_direction_x, line_direction_y, line_direction_z };
    float plane_p1[3] = { plane_p1_x, plane_p1_y, plane_p1_z };
    float plane_p2[3] = { plane_p2_x, plane_p2_y, plane_p2_z };
    float plane_p3[3] = { plane_p3_x, plane_p3_y, plane_p3_z };
    return line_intersect_plane_cuda(line_point, line_direction, plane_p1, plane_p2, plane_p3);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_facet_index_map", &forward_facet_index_map, "FORWARD_FACET_INDEX_MAP (CUDA)");
    m.def("backward_voxel_map", &backward_voxel_map, "BACKWARD_VOXEL_MAP (CUDA)");
    m.def("is_point_in_tetrahedron", &is_point_in_tetrahedron, "IS_POINT_IN_TETRAHEDRON (CUDA)");
    m.def("is_point_in_triangle", &is_point_in_triangle, "IS_POINT_IN_TRIANGLE (CUDA)");
    m.def("is_projected_point_in_triangle", &is_projected_point_in_triangle, "IS_PROJECTED_POINT_IN_TRIANGLE (CUDA)");
    m.def("line_intersect_plane", &line_intersect_plane, "LINE_INTERSECT_PLANE (CUDA)");

}
