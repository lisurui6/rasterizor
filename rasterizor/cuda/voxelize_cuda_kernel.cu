#include <iostream>
#include <ATen/ATen.h>
#include <math.h>   
#include <cuda.h>
#include <cuda_runtime.h>

// for the older gpus atomicAdd with double arguments does not exist

template <typename scalar_t>
__device__ __host__ bool is_point_in_tetrahedron_kernel(
    scalar_t point[3],
    scalar_t vertex_1[3],
    scalar_t vertex_2[3],
    scalar_t vertex_3[3],
    scalar_t vertex_4[3]) {
    //const scalar_t t11 = facet[0] - facet[9];
    //const scalar_t t12 = facet[3] - facet[9];
    //const scalar_t t13 = facet[6] - facet[9];
    //const scalar_t t21 = facet[1] - facet[10];
    //const scalar_t t22 = facet[4] - facet[10];
    //const scalar_t t23 = facet[7] - facet[10];
    //const scalar_t t31 = facet[2] - facet[11];
    //const scalar_t t32 = facet[5] - facet[11];
    //const scalar_t t33 = facet[8] - facet[11];
    //const scalar_t t_inv_11 = t22 * t33 - t23 * t32;
    //const scalar_t t_inv_12 = t13 * t32 - t12 * t33;
    //const scalar_t t_inv_13 = t12 * t23 - t13 * t22;
    //const scalar_t t_inv_21 = t23 * t31 - t21 * t33;
    //const scalar_t t_inv_22 = t11 * t33 - t13 * t31;
    //const scalar_t t_inv_23 = t13 * t21 - t11 * t23;
    //const scalar_t t_inv_31 = t21 * t32 - t22 * t31;
    //const scalar_t t_inv_32 = t12 * t31 - t11 * t32;
    //const scalar_t t_inv_33 = t11 * t22 - t12 * t21;
    //const scalar_t t_det = t11 * (t22 * t33 - t23 * t32) - t12 * (t21 * t33 - t23 * t31) + t13 * (t21 * t32 - t22 * t31);
    //const scalar_t lambda_1 = (t_inv_11 * (xp - facet[9]) + t_inv_12 * (yp - facet[10]) + t_inv_13 * (zp - facet[11])) / t_det;
    //const scalar_t lambda_2 = (t_inv_21 * (xp - facet[9]) + t_inv_22 * (yp - facet[10]) + t_inv_23 * (zp - facet[11])) / t_det;
    //const scalar_t lambda_3 = (t_inv_31 * (xp - facet[9]) + t_inv_32 * (yp - facet[10]) + t_inv_33 * (zp - facet[11])) / t_det;
    //const scalar_t lambda_4 = 1 - lambda_1 - lambda_2 - lambda_3;
    //if (isnan(lambda_1))
    //    continue;
    //if ((lambda_1 < 0) || (lambda_2 < 0) || (lambda_3 < 0) || (lambda_4 < 0) || (lambda_1 > 1) || (lambda_2 > 1) || (lambda_3 > 1) || (lambda_4 > 1))
    //    continue;
    const scalar_t t11 = vertex_1[0] - vertex_4[0];
    const scalar_t t12 = vertex_2[0] - vertex_4[0];
    const scalar_t t13 = vertex_3[0] - vertex_4[0];
    const scalar_t t21 = vertex_1[1] - vertex_4[1];
    const scalar_t t22 = vertex_2[1] - vertex_4[1];
    const scalar_t t23 = vertex_3[1] - vertex_4[1];
    const scalar_t t31 = vertex_1[2] - vertex_4[2];
    const scalar_t t32 = vertex_2[2] - vertex_4[2];
    const scalar_t t33 = vertex_3[2] - vertex_4[2];
    const scalar_t t_inv_11 = t22 * t33 - t23 * t32;
    const scalar_t t_inv_12 = t13 * t32 - t12 * t33;
    const scalar_t t_inv_13 = t12 * t23 - t13 * t22;
    const scalar_t t_inv_21 = t23 * t31 - t21 * t33;
    const scalar_t t_inv_22 = t11 * t33 - t13 * t31;
    const scalar_t t_inv_23 = t13 * t21 - t11 * t23;
    const scalar_t t_inv_31 = t21 * t32 - t22 * t31;
    const scalar_t t_inv_32 = t12 * t31 - t11 * t32;
    const scalar_t t_inv_33 = t11 * t22 - t12 * t21;
    const scalar_t t_det = t11 * (t22 * t33 - t23 * t32) - t12 * (t21 * t33 - t23 * t31) + t13 * (t21 * t32 - t22 * t31);
    const scalar_t lambda_1 = (t_inv_11 * (point[0] - vertex_4[0]) + t_inv_12 * (point[1] - vertex_4[1]) + t_inv_13 * (point[2] - vertex_4[2])) / t_det;
    const scalar_t lambda_2 = (t_inv_21 * (point[0] - vertex_4[0]) + t_inv_22 * (point[1] - vertex_4[1]) + t_inv_23 * (point[2] - vertex_4[2])) / t_det;
    const scalar_t lambda_3 = (t_inv_31 * (point[0] - vertex_4[0]) + t_inv_32 * (point[1] - vertex_4[1]) + t_inv_33 * (point[2] - vertex_4[2])) / t_det;
    const scalar_t lambda_4 = 1 - lambda_1 - lambda_2 - lambda_3;
    if (isnan(lambda_1))
        return false;
    if ((lambda_1 < 0) || (lambda_2 < 0) || (lambda_3 < 0) || (lambda_4 < 0) || (lambda_1 > 1) || (lambda_2 > 1) || (lambda_3 > 1) || (lambda_4 > 1))
        return false;
    return true;
}


template <typename scalar_t>
__device__ __host__ bool is_point_in_triangle_kernel(
    scalar_t point[2], 
    scalar_t vertex_1[2], 
    scalar_t vertex_2[2],
    scalar_t vertex_3[2]
) {
    //scalar_t t_det = (p[0][0] - p[2][0]) * (p[1][1] - p[2][1]) - (p[1][0] - p[2][0]) * (p[0][1] - p[2][1]);
    //scalar_t lambda_1 = ((p[1][1] - p[2][1]) * (d0 - p[2][0]) + (p[2][0] - p[1][0]) * (d1 - p[2][1])) / t_det;
    //scalar_t lambda_2 = ((p[2][1] - p[0][1]) * (d0 - p[2][0]) + (p[0][0] - p[2][0]) * (d1 - p[2][1])) / t_det;
    scalar_t t_det = (vertex_1[0] - vertex_3[0]) * (vertex_2[1] - vertex_3[1]) - (vertex_2[0] - vertex_3[0]) * (vertex_1[1] - vertex_3[1]);
    scalar_t lambda_1 = ((vertex_2[1] - vertex_3[1]) * (point[0] - vertex_3[0]) + (vertex_3[0] - vertex_2[0]) * (point[1] - vertex_3[1])) / t_det;
    scalar_t lambda_2 = ((vertex_3[1] - vertex_1[1]) * (point[0] - vertex_3[0]) + (vertex_1[0] - vertex_3[0]) * (point[1] - vertex_3[1])) / t_det;
    scalar_t lambda_3 = 1 - lambda_1 - lambda_2;
    if ((lambda_1 < 0) || (lambda_2 < 0) || (lambda_3 < 0) || (lambda_1 > 1) || (lambda_2 > 1) || (lambda_3 > 1) || (t_det == 0)) {
        return false;
    }
    return true;
}


template <typename scalar_t>
__device__ __host__ bool is_projected_point_in_triangle_kernel(
    scalar_t point[3],
    scalar_t vertex_1[3],
    scalar_t vertex_2[3],
    scalar_t vertex_3[3]
) {
    // Is the projected point onto the plane T, inside the triangle T?
    // Computing the Barycentric Coordinates of a Projected Point, Wolfgang Heidrich
    scalar_t u[3];
    scalar_t v[3];
    scalar_t w[3];
    scalar_t n[3];
    u[0] = vertex_2[0] - vertex_1[0];
    u[1] = vertex_2[1] - vertex_1[1];
    u[2] = vertex_2[2] - vertex_1[2];

    v[0] = vertex_3[0] - vertex_1[0];
    v[1] = vertex_3[1] - vertex_1[1];
    v[2] = vertex_3[2] - vertex_1[2];

    w[0] = point[0] - vertex_1[0];
    w[1] = point[1] - vertex_1[1];
    w[2] = point[2] - vertex_1[2];

    // n = u X v
    n[0] = u[1] * v[2] - u[2] * v[1];
    n[1] = u[2] * v[0] - u[0] * v[2];
    n[2] = u[0] * v[1] - u[1] * v[0];

    scalar_t n_norm_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    scalar_t uxw[3];
    uxw[0] = u[1] * w[2] - u[2] * w[1];
    uxw[1] = u[2] * w[0] - u[0] * w[2];
    uxw[2] = u[0] * w[1] - u[1] * w[0];
    scalar_t b1;
    // b1 = ( u x w ) . n / (n^2)
    b1 = ( uxw[0] * n[0] + uxw[1] * n[1] + uxw[2] * n[2] ) / n_norm_sq;

    scalar_t wxv[3];
    wxv[0] = w[1] * v[2] - w[2] * v[1];
    wxv[1] = w[2] * v[0] - w[0] * v[2];
    wxv[2] = w[0] * v[1] - w[1] * v[0];
    scalar_t b2 = (wxv[0] * n[0] + wxv[1] * n[1] + wxv[2] * n[2]) / n_norm_sq;

    scalar_t b3 = 1 - b1 - b2;

    if ((b1 < 0) || (b2 < 0) || (b3 < 0) || (b1 > 1) || (b2 > 1) || (b3 > 1)) {
        return false;
    }
    return true;
}

template <typename scalar_t>
__device__ __host__ bool line_intersect_plane_kernel(
    scalar_t cross_point[3],
    scalar_t line_point[3],
    scalar_t line_direction[3],
    scalar_t plane_p1[3],
    scalar_t plane_p2[3],
    scalar_t plane_p3[3]
) {
    scalar_t plane_norm[3];

    plane_norm[0] = (plane_p2[1] - plane_p1[1]) * (plane_p3[2] - plane_p1[2]) - (plane_p2[2] - plane_p1[2]) * (plane_p3[1] - plane_p1[1]);
    plane_norm[1] = (plane_p2[2] - plane_p1[2]) * (plane_p3[0] - plane_p1[0]) - (plane_p2[0] - plane_p1[0]) * (plane_p3[2] - plane_p1[2]);
    plane_norm[2] = (plane_p2[0] - plane_p1[0]) * (plane_p3[1] - plane_p1[1]) - (plane_p2[1] - plane_p1[1]) * (plane_p3[0] - plane_p1[0]);

    scalar_t denominator = (line_direction[0] * plane_norm[0] + line_direction[1] * plane_norm[1] + line_direction[2] * plane_norm[2]);
    if (denominator == 0) {
        return false;
    }
    scalar_t d = ((plane_p1[0] - line_point[0]) * plane_norm[0] + (plane_p1[1] - line_point[1]) * plane_norm[1] + (plane_p1[2] - line_point[2]) * plane_norm[2]) / denominator;

    cross_point[0] = line_point[0] + line_direction[0] * d;
    cross_point[1] = line_point[1] + line_direction[1] * d;
    cross_point[2] = line_point[2] + line_direction[2] * d;
    return true;
}


__device__ __host__ int to_map_index_kernel(
    int bn,
    int x,
    int y,
    int z,
    int voxel_width,
    int voxel_depth,
    int voxel_height
) {
    int map_index = bn * voxel_width * voxel_height * voxel_depth + x * voxel_height * voxel_depth + y * voxel_height + z;
    return map_index;
};


template <typename scalar_t>
__global__ void forward_facet_index_map_cuda_kernel_2(
    const scalar_t* facets,
    int32_t* __restrict__ facet_index_map,
    int batch_size,
    int num_facets,
    int voxel_width,
    int voxel_depth,
    int voxel_height) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * voxel_width * voxel_height * voxel_depth) {
        return;
    }
    const int batch_num = i / (voxel_width * voxel_height * voxel_depth);
    const int voxel_num = i % (voxel_width * voxel_height * voxel_depth);
    const int xi = voxel_num / (voxel_depth * voxel_height);
    const int yi = (voxel_num % (voxel_depth * voxel_height)) / voxel_height;
    const int zi = voxel_num % (voxel_depth * voxel_height) % voxel_height;
    const scalar_t zp = (2. * zi + 1 - voxel_height) / voxel_height;
    const scalar_t yp = (2. * yi + 1 - voxel_depth) / voxel_depth;
    const scalar_t xp = (2. * xi + 1 - voxel_width) / voxel_width;
    scalar_t point[3] = { xp, yp, zp };

    const scalar_t* facet = &facets[batch_num * num_facets * 12] - 12;
    int facet_index_min = -1;
    for (int facet_num = 0; facet_num < num_facets; facet_num++) {
        /* go to next facet */
        facet += 12;

        /* return if backside */
        // if ((face[5] - face[1]) * (face[2] - face[0]) < (face[3] - face[1]) * (face[4] - face[0]))
        //    continue;

        /* check [py, px] is inside the face */

        scalar_t vertex_1[3] = { facet[0], facet[1], facet[2] };
        scalar_t vertex_2[3] = { facet[3], facet[4], facet[5] };
        scalar_t vertex_3[3] = { facet[6], facet[7], facet[8] };
        scalar_t vertex_4[3] = { facet[9], facet[10], facet[11] };
        if (is_point_in_tetrahedron_kernel(point, vertex_1, vertex_2, vertex_3, vertex_4)) {
            facet_index_min = facet_num;
        }
        else {
            continue;
        }
        break;
    }

    /* set to global memory */
    if (0 <= facet_index_min) {
        facet_index_map[i] = facet_index_min;
    }
}


template <typename scalar_t>
__global__ void backward_voxel_map_cuda_kernel(
    const scalar_t* facets,
    int32_t* facet_index_map,
    scalar_t* alpha_map,
    scalar_t* grad_alpha_map,
    scalar_t* grad_facets,
    size_t batch_size,
    size_t num_facets,
    int voxel_width,
    int voxel_depth,
    int voxel_height,
    scalar_t eps,
    scalar_t eps_in) {
    const int ik = blockIdx.x * blockDim.x + threadIdx.x;
    if (ik >= batch_size * num_facets) {
        return;
    }
    const int bn = ik / num_facets;
    const int fn = ik % num_facets;
    const scalar_t* facet = &facets[ik * 12];
    scalar_t grad_facet[12] = {};
    ///* check backside */
    //if ((facet[5] - facet[1]) * (facet[2] - facet[0]) < (facet[3] - facet[1]) * (facet[4] - facet[0]))
    //    return;

    /* for each face*/
    scalar_t out_grad = 0;
    scalar_t in_grad = 0;
    //scalar_t eps_in = 0.5;
    for (int face_num = 0; face_num < 4; face_num++) {
        int pi[4];
        scalar_t pp[4][3]; // arrange points according to the current face

        int dims[3] = { voxel_width, voxel_depth, voxel_height }; // x_size, y_size, z_size

        for (int num = 0; num < 4; num++)
            pi[num] = (face_num + num) % 4;
        for (int num = 0; num < 4; num++) {
            for (int dim = 0; dim < 3; dim++) {
                pp[num][dim] = 0.5 * (facet[3 * pi[num] + dim] * dims[dim] + dims[dim] - 1);
            }
        }
        const scalar_t t_det = (pp[1][1] - pp[2][1]) * (pp[0][0] - pp[2][0]) + (pp[2][0] - pp[1][0]) * (pp[0][1] - pp[2][1]);
        //if (t_det == 0)
        //    continue;

        /* for dz, dx, dy */
        
        for (int axis = 0; axis < 3; axis++) {
            scalar_t p[4][3];  // move pp's axis.
            for (int num = 0; num < 4; num++) {
                for (int dim = 0; dim < 3; dim++) {
                    p[num][dim] = pp[num][(dim + axis) % 3];
                }
            }

            /* set direction */
            /* along face */
            int d0_from, d0_to;
            d0_from = max(ceil(min(min(p[0][0], p[1][0]), p[2][0])), 0.);
            d0_to = min(max(max(p[0][0], p[1][0]), p[2][0]), dims[axis] - 1.);

            int d1_from, d1_to;
            d1_from = max(ceil(min(min(p[0][1], p[1][1]), p[2][1])), 0.);
            d1_to = min(max(max(p[0][1], p[1][1]), p[2][1]), dims[(axis + 1) % 3] - 1.);
            //if (bn == 0)
            //    printf("bn %d, face %d, axis %d, d0_from %d, d0_to %d, d1_from %d, d1_to %d\n", bn, face_num, axis, d0_from, d0_to, d1_from, d1_to);
            int n_inside = 0;
            for (int d0 = d0_from; d0 <= d0_to; d0++) {
                for (int d1 = d1_from; d1 <= d1_to; d1++) {
                    int d2_from = 0;
                    int d2_to = 0;
                    
                    // check if d0, d1 is in the projected triangle (p[0][0], p[0][1]), (p[1][0], p[1][1]), (p[2][0], p[2][1])
                    scalar_t point[2] = { d0, d1 };
                    scalar_t v1[2] = { p[0][0], p[0][1] };
                    scalar_t v2[2] = { p[1][0], p[1][1] };
                    scalar_t v3[2] = { p[2][0], p[2][1] };

                    if (!is_point_in_triangle_kernel(point, v1, v2, v3)) {
                        continue;
                    }

                    /* get cross point */
                    scalar_t plane_p1[3] = { pp[0][0], pp[0][1], pp[0][2] };
                    scalar_t plane_p2[3] = { pp[1][0], pp[1][1], pp[1][2] };
                    scalar_t plane_p3[3] = { pp[2][0], pp[2][1], pp[2][2] };

                    scalar_t l0[3];
                    l0[(2 + axis) % 3] = 0;
                    l0[(0 + axis) % 3] = d0;
                    l0[(1 + axis) % 3] = d1;
                    scalar_t l[3] = { 0, 0, 0 };
                    l[(2 + axis) % 3] = 1;
                    scalar_t p_cross[3];

                    bool flag = line_intersect_plane_kernel(p_cross, l0, l, plane_p1, plane_p2, plane_p3);
                    if (!flag) {
                        printf("skip, line and plane parallel.\n");
                        continue;
                    }
                    int d2_in, d2_out;
                    scalar_t d2_cross = p_cross[(2 + axis) % 3];
                    // iterate over all the faces to see how many l cross with that is lower than d2_cross
                    int num_extra_cross = 0;
                    int direction = 1;
                    for (int i = 1; i < 4; i++) {
                        scalar_t plane_p1[3] = { pp[(0 + i) % 4][0], pp[(0 + i) % 4][1], pp[(0 + i) % 4][2] };
                        scalar_t plane_p2[3] = { pp[(1 + i) % 4][0], pp[(1 + i) % 4][1], pp[(1 + i) % 4][2] };
                        scalar_t plane_p3[3] = { pp[(2 + i) % 4][0], pp[(2 + i) % 4][1], pp[(2 + i) % 4][2] };
                        scalar_t p_cross[3];
                        flag = line_intersect_plane_kernel(p_cross, l0, l, plane_p1, plane_p2, plane_p3);
                        if (!flag) {
                            continue;
                        }
                        if ((p_cross[(2 + axis) % 3] > d2_cross) && (is_projected_point_in_triangle_kernel(p_cross, plane_p1, plane_p2, plane_p3))) {
                            direction = -1;
                            break;
                        }
                    }

                    if (0 < direction)
                        d2_in = floor(d2_cross);
                    else
                        d2_in = ceil(d2_cross);
                    d2_out = d2_in + direction;
                    /* continue if cross point is not shown */
                    if (d2_in < 0 || dims[(2 + axis) % 3] <= d2_in)
                        continue;
                    if (d2_out < 0 || dims[(2 + axis) % 3] <= d2_out)
                        continue;
                    /* get color of in-pixel and out-pixel */
                    scalar_t alpha_in;
                    scalar_t alpha_out;

                    int map_index_in, map_index_out;
                    if (axis == 0) {
                        map_index_in = to_map_index_kernel(bn, d0, d1, d2_in, voxel_width, voxel_depth, voxel_height);
                        map_index_out = to_map_index_kernel(bn, d0, d1, d2_out, voxel_width, voxel_depth, voxel_height);
                    }
                    else if (axis == 1) {
                        map_index_in = to_map_index_kernel(bn, d2_in, d0, d1, voxel_width, voxel_depth, voxel_height);
                        map_index_out = to_map_index_kernel(bn, d2_out, d0, d1, voxel_width, voxel_depth, voxel_height);
                    }
                    else {
                        map_index_in = to_map_index_kernel(bn, d1, d2_in, d0, voxel_width, voxel_depth, voxel_height);
                        map_index_out = to_map_index_kernel(bn, d1, d2_out, d0, voxel_width, voxel_depth, voxel_height);
                    }
                    alpha_in = alpha_map[map_index_in];
                    alpha_out = alpha_map[map_index_out];
                    
                    /* out */
                    bool is_in_fn = (facet_index_map[map_index_in] == fn);
                    //bool is_in_fn = true;
                    int d2_limit;
                    if (0 < direction)
                        d2_limit = dims[axis] - 1;
                    else
                        d2_limit = 0;

                    d2_from = max(min(d2_out, d2_limit), 0);
                    d2_to = min(max(d2_out, d2_limit), dims[(axis + 2) % 3] - 1);

                    
                    if (is_in_fn) {
                        scalar_t* alpha_map_p;
                        scalar_t* grad_alpha_map_p;
                        int map_offset, map_index_from;
                        if (axis == 0) {
                            map_offset = 1;
                            map_index_from = to_map_index_kernel(bn, d0, d1, d2_from, voxel_width, voxel_depth, voxel_height);
                        }
                        else if (axis == 1) {
                            map_offset = voxel_height * voxel_depth;
                            map_index_from = to_map_index_kernel(bn, d2_from, d0, d1, voxel_width, voxel_depth, voxel_height);

                        }
                        else {
                            map_offset = voxel_height;
                            map_index_from = to_map_index_kernel(bn, d1, d2_from, d0, voxel_width, voxel_depth, voxel_height);

                        }

                        alpha_map_p = &alpha_map[map_index_from];
                        grad_alpha_map_p = &grad_alpha_map[map_index_from];
                        
                        //if (bn == 0)
                        //    printf("face %d, axis %d, d0 %d, d1 %d, d2_from %d, d2_to %d\n", face_num, axis, d0, d1, d2_from, d2_to);
                        
                        for (int d2 = d2_from; d2 <= d2_to; d2++) {
                            scalar_t grad_alpha_p = *grad_alpha_map_p;  //negative if true is 1, pred is 0
                            scalar_t alpha_p = *alpha_map_p;
                            scalar_t diff_grad = (alpha_p - 1) * grad_alpha_p;  //(*alpha_map_p - alpha_in) negative or zero
                            //if ((bn == 0) && ((grad_alpha_p > 1) || (grad_alpha_p < -1)))
                            //    printf("out face %d, axis %d, d0 %d, d1 %d, d2 %d, grad_alpha %f, diff_grad %f, alpha p %f, alpha_in %f\n", face_num, axis, d0, d1, d2, grad_alpha_p, diff_grad, alpha_p, alpha_in);
                            alpha_map_p += map_offset;
                            grad_alpha_map_p += map_offset;
                            scalar_t dp[3] = { d0, d1, d2 };
                            if (diff_grad <= 0) {
                                continue;
                            }
                            scalar_t l[3] = { 0, 0, 0 };
                            l[(2 + axis) % 3] = direction;

                            // grad for pp[0] (pi[0])
                            {
                                scalar_t l0[3] = { pp[0][0], pp[0][1], pp[0][2] };
                                scalar_t plane_p1[3] = { pp[1][0], pp[1][1], pp[1][2] };
                                scalar_t plane_p2[3] = { pp[2][0], pp[2][1], pp[2][2] };
                                scalar_t plane_p3[3] = { dp[(3 - axis) % 3], dp[(4 - axis) % 3], dp[(5 - axis) % 3] };
                                scalar_t p_cross[3];

                                // flag is true if line and plane are not parallel. 
                                bool flag = line_intersect_plane_kernel(p_cross, l0, l, plane_p1, plane_p2, plane_p3);

                                if (flag) {
                                    scalar_t dist = direction * sqrt((p_cross[0] - l0[0]) * (p_cross[0] - l0[0]) + (p_cross[1] - l0[1]) * (p_cross[1] - l0[1]) + (p_cross[2] - l0[2]) * (p_cross[2] - l0[2]));
                                    dist = dist * 2 / dims[(axis + 2) % 3];
                                    dist = (0 < dist) ? dist + eps : dist - eps;
                                    grad_facet[pi[0] * 3 + (2 + axis) % 3] -= diff_grad / dist;
                                    //if ((bn == 0) && ((grad_alpha_p > 1) || (grad_alpha_p < -1)))
                                    //    printf("out face %d, axis %d, d0 %d, d1 %d, d2 %d, grad_alpha %f, diff_grad %f, dist %f\n", face_num, axis, d0, d1, d2, grad_alpha_p, diff_grad, dist);
                                    if (diff_grad / dist > 0)
                                        out_grad += diff_grad / dist;
                                    else
                                        out_grad -= diff_grad / dist;
                                    
                                }
                            }
                            // grad for pp[1] (pi[1])
                            {
                                scalar_t l0[3] = { pp[1][0], pp[1][1], pp[1][2] };

                                scalar_t plane_p1[3] = { pp[0][0], pp[0][1], pp[0][2] };
                                scalar_t plane_p2[3] = { pp[2][0], pp[2][1], pp[2][2] };
                                scalar_t plane_p3[3] = { dp[(3 - axis) % 3], dp[(4 - axis) % 3], dp[(5 - axis) % 3] };
                                scalar_t p_cross[3];

                                bool flag = line_intersect_plane_kernel(p_cross, l0, l, plane_p1, plane_p2, plane_p3);

                                if (flag) {
                                    scalar_t dist = direction * sqrt((p_cross[0] - l0[0]) * (p_cross[0] - l0[0]) + (p_cross[1] - l0[1]) * (p_cross[1] - l0[1]) + (p_cross[2] - l0[2]) * (p_cross[2] - l0[2]));
                                    dist = dist * 2 / dims[(axis + 2) % 3];
                                    dist = (0 < dist) ? dist + eps : dist - eps;
                                    grad_facet[pi[1] * 3 + (2 + axis) % 3] -= diff_grad / dist;
                                    if (diff_grad / dist > 0)
                                        out_grad += diff_grad / dist;
                                    else
                                        out_grad -= diff_grad / dist;
                                    
                                    //if ((bn == 0) && ((grad_alpha_p > 1) || (grad_alpha_p < -1)))
                                    //    printf("out face %d, axis %d, d0 %d, d1 %d, d2 %d, grad_alpha %f, diff_grad %f, dist %f\n", face_num, axis, d0, d1, d2, grad_alpha_p, diff_grad, dist);
                                }
                            }
                            // grad for pp[2] (pi[2])
                            {
                                scalar_t l0[3] = { pp[2][0], pp[2][1], pp[2][2] };
                                scalar_t plane_p1[3] = { pp[1][0], pp[1][1], pp[1][2] };
                                scalar_t plane_p2[3] = { pp[0][0], pp[0][1], pp[0][2] };
                                scalar_t plane_p3[3] = { dp[(3 - axis) % 3], dp[(4 - axis) % 3], dp[(5 - axis) % 3] };
                                scalar_t p_cross[3];

                                bool flag = line_intersect_plane_kernel(p_cross, l0, l, plane_p1, plane_p2, plane_p3);
                                if (flag) {
                                    scalar_t dist = direction * sqrt((p_cross[0] - l0[0]) * (p_cross[0] - l0[0]) + (p_cross[1] - l0[1]) * (p_cross[1] - l0[1]) + (p_cross[2] - l0[2]) * (p_cross[2] - l0[2]));
                                    dist = dist * 2 / dims[(axis + 2) % 3];
                                    dist = (0 < dist) ? dist + eps : dist - eps;
                                    grad_facet[pi[2] * 3 + (2 + axis) % 3] -= diff_grad / dist;
                                    if (diff_grad / dist > 0)
                                        out_grad += diff_grad / dist;
                                    else
                                        out_grad -= diff_grad / dist;
                                    //if ((bn == 0) && ((grad_alpha_p > 1) || (grad_alpha_p < -1)))
                                    //    printf("out face %d, axis %d, d0 %d, d1 %d, d2 %d, grad_alpha %f, diff_grad %f, dist %f\n", face_num, axis, d0, d1, d2, grad_alpha_p, diff_grad, dist);
                                }
                            }
                        }
                    }
                    //printf("outgrad %f\n", out_grad);
                    /* in */
                    {
                        
                        int d2_limit;
                        scalar_t d2_cross2;
                        // get the other cross point
                        scalar_t l0[3];
                        l0[(2 + axis) % 3] = 0;
                        l0[(0 + axis) % 3] = d0;
                        l0[(1 + axis) % 3] = d1;
                        scalar_t l[3] = { 0, 0, 0 };
                        l[(2 + axis) % 3] = 1;
                        for (int i = 1; i < 4; i++) {
                            scalar_t plane_p1[3] = { pp[(0 + i) % 4][0], pp[(0 + i) % 4][1], pp[(0 + i) % 4][2] };
                            scalar_t plane_p2[3] = { pp[(1 + i) % 4][0], pp[(1 + i) % 4][1], pp[(1 + i) % 4][2] };
                            scalar_t plane_p3[3] = { pp[(2 + i) % 4][0], pp[(2 + i) % 4][1], pp[(2 + i) % 4][2] };
                            scalar_t p_cross[3];
                            flag = line_intersect_plane_kernel(p_cross, l0, l, plane_p1, plane_p2, plane_p3);

                            if (!flag) {
                                continue;
                            }
                            if (is_projected_point_in_triangle_kernel(p_cross, plane_p1, plane_p2, plane_p3)) {
                                d2_cross2 = p_cross[(2 + axis) % 3];
                                break;
                            }

                        }

                        if (0 < direction)
                            d2_limit = ceil(d2_cross2);
                        else
                            d2_limit = floor(d2_cross2);
                        int d2_from = max(min(d2_in, d2_limit), 0);
                        int d2_to = min(max(d2_in, d2_limit), dims[(axis + 2) % 3] - 1);

                        int* facet_index_map_p;
                        scalar_t* alpha_map_p;
                        scalar_t* grad_alpha_map_p;
                        int map_index_from;
                        int map_offset;
                        if (axis == 0) {
                            map_offset = 1;
                            map_index_from = to_map_index_kernel(bn, d0, d1, d2_from, voxel_width, voxel_depth, voxel_height);
                        }
                        else if (axis == 1) {
                            map_offset = voxel_height * voxel_depth;
                            map_index_from = to_map_index_kernel(bn, d2_from, d0, d1, voxel_width, voxel_depth, voxel_height);
                        }
                        else {
                            map_offset = voxel_height;
                            map_index_from = to_map_index_kernel(bn, d1, d2_from, d0, voxel_width, voxel_depth, voxel_height);
                        }
                        facet_index_map_p = &facet_index_map[map_index_from] - map_offset;

                        alpha_map_p = &alpha_map[map_index_from] - map_offset;
                        grad_alpha_map_p = &grad_alpha_map[map_index_from] - map_offset;
                        //if (bn == 0)
                        //    printf("face %d, axis %d, d0 %d, d1 %d, d2_from %d, d2_to %d, d2_in %d, direction %d\n", face_num, axis, d0, d1, d2_from, d2_to, d2_in, direction);
                        for (int d2 = d2_from; d2 <= d2_to; d2++) {
                            facet_index_map_p += map_offset;

                            alpha_map_p += map_offset;
                            grad_alpha_map_p += map_offset;

                            if (*facet_index_map_p != fn)
                                continue;

                            scalar_t diff_grad = 0;
                            diff_grad = (*alpha_map_p - 0) * *grad_alpha_map_p;  // grad_alpha positive if true is 0, pred is 1
                            scalar_t dp[3] = { d0, d1, d2 };
                            //if ((bn == 0) && ((*grad_alpha_map_p > 1) || (*grad_alpha_map_p < -1)))
                            //    printf("in face %d, axis %d, d0 %d, d1 %d, d2 %d, grad_alpha %f, diff_grad %f, alpha p %f, alpha_out %f\n", face_num, axis, d0, d1, d2, *grad_alpha_map_p, diff_grad, *alpha_map_p, alpha_out);
                            if (diff_grad <= 0)
                                continue;


                            // grad for pp[0] (pi[0])
                            {
                                scalar_t l0[3] = { pp[0][0], pp[0][1], pp[0][2] };
                                scalar_t plane_p1[3] = { pp[1][0], pp[1][1], pp[1][2] };
                                scalar_t plane_p2[3] = { pp[2][0], pp[2][1], pp[2][2] };
                                scalar_t plane_p3[3] = { dp[(3 - axis) % 3], dp[(4 - axis) % 3], dp[(5 - axis) % 3] };
                                scalar_t p_cross[3];
                                scalar_t l[3] = { 0, 0, 0 };
                                l[(2 + axis) % 3] = -direction;
                                bool flag = line_intersect_plane_kernel(p_cross, l0, l, plane_p1, plane_p2, plane_p3);

                                if (flag) {
                                    scalar_t dist = (-direction) * sqrt((p_cross[0] - l0[0]) * (p_cross[0] - l0[0]) + (p_cross[1] - l0[1]) * (p_cross[1] - l0[1]) + (p_cross[2] - l0[2]) * (p_cross[2] - l0[2]));
                                    if (dist >= 0) {
                                        dist = max(dist, eps_in);
                                    }
                                    else {
                                        dist = min(dist, -eps_in);
                                    }
                                    //dist = (0 < dist) ? dist + eps_in : dist - eps_in;
                                    dist = dist * 2 / dims[(axis + 2) % 3];
                                    //// dist = (0 < dist) ? dist + eps_in : dist - eps_in;
                                    grad_facet[pi[0] * 3 + (2 + axis) % 3] -= diff_grad / dist;


                                    //if (((dist > 0) && (dist > eps_in)) || ((dist < 0) && (dist < -eps_in))) {
                                    //    //dist = dist * 2 / dims[(axis + 2) % 3];
                                    //    // dist = (0 < dist) ? dist + eps_in : dist - eps_in;
                                    //    grad_facet[pi[0] * 3 + (2 + axis) % 3] -= diff_grad / dist;
                                    //    if (diff_grad / dist > 0)
                                    //        in_grad += diff_grad / dist;
                                    //    else
                                    //        in_grad -= diff_grad / dist;

                                    //}
                                    

                                }
                            }
                            // grad for pp[1] (pi[1])
                            {
                                scalar_t l0[3] = { pp[1][0], pp[1][1], pp[2][2] };
                                scalar_t plane_p1[3] = { pp[0][0], pp[0][1], pp[0][2] };
                                scalar_t plane_p2[3] = { pp[2][0], pp[2][1], pp[2][2] };
                                scalar_t plane_p3[3] = { dp[(3 - axis) % 3], dp[(4 - axis) % 3], dp[(5 - axis) % 3] };
                                scalar_t p_cross[3];
                                scalar_t l[3] = { 0, 0, 0 };
                                l[(2 + axis) % 3] = -direction;
                                bool flag = line_intersect_plane_kernel(p_cross, l0, l, plane_p1, plane_p2, plane_p3);

                                if (flag) {
                                    scalar_t dist = (-direction) * sqrt((p_cross[0] - l0[0]) * (p_cross[0] - l0[0]) + (p_cross[1] - l0[1]) * (p_cross[1] - l0[1]) + (p_cross[2] - l0[2]) * (p_cross[2] - l0[2]));
                                    if (dist >= 0) {
                                        dist = max(dist, eps_in);
                                    }
                                    else {
                                        dist = min(dist, -eps_in);
                                    }
                                    //dist = (0 < dist) ? dist + eps_in : dist - eps_in;
                                    dist = dist * 2 / dims[(axis + 2) % 3];
                                    // dist = (0 < dist) ? dist + eps_in : dist - eps_in;
                                    grad_facet[pi[1] * 3 + (2 + axis) % 3] -= diff_grad / dist;

                                    //if (((dist > 0) && (dist > eps_in)) || ((dist < 0) && (dist < -eps_in))) {
                                    //    //dist = dist * 2 / dims[(axis + 2) % 3];
                                    //    // dist = (0 < dist) ? dist + eps_in : dist - eps_in;
                                    //    grad_facet[pi[1] * 3 + (2 + axis) % 3] -= diff_grad / dist;
                                    //    if (diff_grad / dist > 0)
                                    //        in_grad += diff_grad / dist;
                                    //    else
                                    //        in_grad -= diff_grad / dist;
                                    //}
                                }
                            }
                            // grad for pp[2] (pi[2])
                            {
                                scalar_t l0[3] = { pp[2][0], pp[2][1], pp[2][2] };
                                scalar_t plane_p1[3] = { pp[1][0], pp[1][1], pp[1][2] };
                                scalar_t plane_p2[3] = { pp[0][0], pp[0][1], pp[0][2] };
                                scalar_t plane_p3[3] = { dp[(3 - axis) % 3], dp[(4 - axis) % 3], dp[(5 - axis) % 3] };
                                scalar_t p_cross[3];
                                scalar_t l[3] = { 0, 0, 0 };
                                l[(2 + axis) % 3] = -direction;
                                bool flag = line_intersect_plane_kernel(p_cross, l0, l, plane_p1, plane_p2, plane_p3);

                                if (flag) {
                                    scalar_t dist = (-direction) * sqrt((p_cross[0] - l0[0]) * (p_cross[0] - l0[0]) + (p_cross[1] - l0[1]) * (p_cross[1] - l0[1]) + (p_cross[2] - l0[2]) * (p_cross[2] - l0[2]));
                                    if (dist >= 0) {
                                        dist = max(dist, eps_in);
                                    }
                                    else {
                                        dist = min(dist, -eps_in);
                                    }
                                    //// dist = (0 < dist) ? dist + eps_in : dist - eps_in;
                                    //dist = (0 < dist) ? dist + eps_in : dist - eps_in;
                                    dist = dist * 2 / dims[(axis + 2) % 3];
                                    grad_facet[pi[2] * 3 + (2 + axis) % 3] -= diff_grad / dist;

                                    //if (((dist > 0) && (dist > eps_in)) || ((dist < 0) && (dist < -eps_in))) {
                                    //    //dist = dist * 2 / dims[(axis + 2) % 3];
                                    //    //dist = (0 < dist) ? dist + eps_in : dist - eps_in;
                                    //    grad_facet[pi[2] * 3 + (2 + axis) % 3] -= diff_grad / dist;
                                    //    if (diff_grad / dist > 0)
                                    //        in_grad += diff_grad / dist;
                                    //    else
                                    //        in_grad -= diff_grad / dist;
                                    //}

                                }
                            }

                            /*if ((bn == 0))
                            {
                                printf("in2 face %d, axis %d, d0 %d, d1 %d, d2 %d, grad1 %f, grad4 %f, grad7 %f, grad10 %f\n", face_num, axis, d0, d1, d2, grad_facet[1], grad_facet[4], grad_facet[7], grad_facet[10]);

                            }*/

                        }
                    }
                    
                }
            }
        }
    }
    //printf("outgrad %f\n", out_grad);
    //printf("ingrad %f\n", in_grad);
    /* set to global gradient variable */
    for (int k = 0; k < 12; k++)
        grad_facets[ik * 12 + k] = grad_facet[k];
}



at::Tensor forward_facet_index_map_cuda(
    at::Tensor facets,
    at::Tensor facet_index_map,
    int voxel_width,
    int voxel_depth,
    int voxel_height) {

    const auto batch_size = facets.size(0);
    const auto num_facets = facets.size(1);
    const int threads = 512;

    const dim3 blocks_2((batch_size * voxel_width * voxel_height * voxel_depth - 1) / threads + 1);
    AT_DISPATCH_FLOATING_TYPES(facets.type(), "forward_facet_index_map_cuda_2", ([&] {
        forward_facet_index_map_cuda_kernel_2<scalar_t> << <blocks_2, threads >> > (
            facets.data<scalar_t>(),
            facet_index_map.data<int32_t>(),
            (int)batch_size,
            (int)num_facets,
            (int)voxel_width,
            (int)voxel_depth,
            (int)voxel_height);
        }));
    cudaError_t err = cudaGetLastError();
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in forward_facet_index_map_2: %s\n", cudaGetErrorString(err));
    return facet_index_map;
}


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
    float eps_in) {

    const auto batch_size = facets.size(0);
    const auto num_facets = facets.size(1);
    const int threads = 128;
    const dim3 blocks((batch_size * num_facets - 1) / threads + 1);
    //printf("num facets %d, batch_size %d\n", num_facets, batch_size);

    AT_DISPATCH_FLOATING_TYPES(facets.type(), "backward_voxel_map_cuda", ([&] {
        backward_voxel_map_cuda_kernel<scalar_t> << <blocks, threads >> > (
            facets.data<scalar_t>(),
            facet_index_map.data<int32_t>(),
            alpha_map.data<scalar_t>(),
            grad_alpha_map.data<scalar_t>(),
            grad_facets.data<scalar_t>(),
            batch_size,
            num_facets,
            (int)voxel_width,
            (int)voxel_depth,
            (int)voxel_height,
            (scalar_t)eps,
            (scalar_t)eps_in);
        }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in backward_voxel_map: %s\n", cudaGetErrorString(err));

    return grad_facets;
}


bool is_point_in_tetrahedron_cuda(
    float point[3],
    float vertex_1[3],
    float vertex_2[3],
    float vertex_3[3],
    float vertex_4[3]
) {
    return is_point_in_tetrahedron_kernel(point, vertex_1, vertex_2, vertex_3, vertex_4);
}


bool is_point_in_triangle_cuda(
    float point[2],
    float vertex_1[2],
    float vertex_2[2],
    float vertex_3[2]
) {
    return is_point_in_triangle_kernel(point, vertex_1, vertex_2, vertex_3);
}

template <class T>
void print(T& c) {
    for (typename T::iterator i = c.begin(); i != c.end(); i++) {
        std::cout << *i << std::endl;
    }
}

std::vector<float> line_intersect_plane_cuda(
    float line_point[3],
    float line_direction[3],
    float plane_p1[3],
    float plane_p2[3],
    float plane_p3[3]
) {
    float x[3];
    line_intersect_plane_kernel(x, line_point, line_direction, plane_p1, plane_p2, plane_p3);
    std::vector<float> v(x, x+3);
    return v;
}


bool is_projected_point_in_triangle_cuda(
    float point[3],
    float vertex_1[3],
    float vertex_2[3],
    float vertex_3[3]
) {
    return is_projected_point_in_triangle_kernel(point, vertex_1, vertex_2, vertex_3);
}
