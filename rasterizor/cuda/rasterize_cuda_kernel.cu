#include <iostream>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

// for the older gpus atomicAdd with double arguments does not exist



template <typename scalar_t>
__global__ void forward_face_index_map_cuda_kernel_2(
        const scalar_t* faces,
        int32_t* __restrict__ face_index_map,
        int batch_size,
        int num_faces,
        int image_size) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int is = image_size;
    const int nf = num_faces;
    const int bn = i / (is * is);
    const int pn = i % (is * is);
    const int yi = pn / is;
    const int xi = pn % is;
    const scalar_t yp = (2. * yi + 1 - is) / is;
    const scalar_t xp = (2. * xi + 1 - is) / is;

    const scalar_t* face = &faces[bn * nf * 6] - 6;
    int face_index_min = -1;
    for (int fn = 0; fn < nf; fn++) {
        /* go to next face */
        face += 6;

        /* return if backside */
        if ((face[5] - face[1]) * (face[2] - face[0]) < (face[3] - face[1]) * (face[4] - face[0]))
            continue;

        /* check [py, px] is inside the face */
        if (((yp - face[1]) * (face[2] - face[0]) < (xp - face[0]) * (face[3] - face[1])) ||
            ((yp - face[3]) * (face[4] - face[2]) < (xp - face[2]) * (face[5] - face[3])) ||
            ((yp - face[5]) * (face[0] - face[4]) < (xp - face[4]) * (face[1] - face[5])))
            continue;
        face_index_min = fn;
        break;
    }

    /* set to global memory */
    if (0 <= face_index_min) {
        face_index_map[i] = face_index_min;
    }
}


template <typename scalar_t>
__global__ void backward_pixel_map_cuda_kernel(
		const scalar_t* faces,
        int32_t*  face_index_map,
        scalar_t*  alpha_map,
        scalar_t*  grad_alpha_map,
        scalar_t*  grad_faces,
        size_t batch_size,
        size_t num_faces,
        int image_size,
        scalar_t eps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * num_faces) {
        return;
    }
    const int bn = i / num_faces;
    const int fn = i % num_faces;
    const int is = image_size;
    const scalar_t* face = &faces[i * 6];
    scalar_t grad_face[6] = {};

    /* check backside */
    if ((face[5] - face[1]) * (face[2] - face[0]) < (face[3] - face[1]) * (face[4] - face[0]))
        return;

    /* for each edge */
    for (int edge_num = 0; edge_num < 3; edge_num++) {
        /* set points of target edge */
        int pi[3];
        scalar_t pp[3][2];
        for (int num = 0; num < 3; num++)
            pi[num] = (edge_num + num) % 3;
        for (int num = 0; num < 3; num++) {
            for (int dim = 0; dim < 2; dim++) {
                pp[num][dim] = 0.5 * (face[2 * pi[num] + dim] * is + is - 1);
            }
        }

        /* for dy, dx */
        for (int axis = 0; axis < 2; axis++) {
            /* */
            scalar_t p[3][2];
            for (int num = 0; num < 3; num++) {
                for (int dim = 0; dim < 2; dim++) {
                    p[num][dim] = pp[num][(dim + axis) % 2];
                }
            }

            /* set direction */
            int direction;
            if (axis == 0) {
                if (p[0][0] < p[1][0])
                    direction = -1;
                else
                    direction = 1;
            } else {
                if (p[0][0] < p[1][0])
                    direction = 1;
                else
                    direction = -1;
            }

            /* along edge */
            int d0_from, d0_to;
            d0_from = max(ceil(min(p[0][0], p[1][0])), 0.);
            d0_to = min(max(p[0][0], p[1][0]), is - 1.);
            for (int d0 = d0_from; d0 <= d0_to; d0++) {
                /* get cross point */
                int d1_in, d1_out;
                const scalar_t d1_cross = (p[1][1] - p[0][1]) / (p[1][0] - p[0][0]) * (d0 - p[0][0]) + p[0][1];
                if (0 < direction)
                    d1_in = floor(d1_cross);
                else
                    d1_in = ceil(d1_cross);
                d1_out = d1_in + direction;

                /* continue if cross point is not shown */
                if (d1_in < 0 || is <= d1_in)
                    continue;
                if (d1_out < 0 || is <= d1_out)
                    continue;

                /* get color of in-pixel and out-pixel */
                scalar_t alpha_in;
                scalar_t alpha_out;

                int map_index_in, map_index_out;
                if (axis == 0) {
                    map_index_in = bn * is * is + d1_in * is + d0;
                    map_index_out = bn * is * is + d1_out * is + d0;
                }
                else {
                    map_index_in = bn * is * is + d0 * is + d1_in;
                    map_index_out = bn * is * is + d0 * is + d1_out;
                }
                alpha_in = alpha_map[map_index_in];
                alpha_out = alpha_map[map_index_out];

                /* out */
                bool is_in_fn = (face_index_map[map_index_in] == fn);
                if (is_in_fn) {
                    int d1_limit;
                    if (0 < direction)
                        d1_limit = is - 1;
                    else
                        d1_limit = 0;
                    int d1_from = max(min(d1_out, d1_limit), 0);
                    int d1_to = min(max(d1_out, d1_limit), is - 1);
                    scalar_t* alpha_map_p;
                    scalar_t* grad_alpha_map_p;
                    int map_offset, map_index_from;
                    if (axis == 0) {
                        map_offset = is;
                        map_index_from = bn * is * is + d1_from * is + d0;
                    }
                    else {
                        map_offset = 1;
                        map_index_from = bn * is * is + d0 * is + d1_from;
                    }

                    alpha_map_p = &alpha_map[map_index_from];
                    grad_alpha_map_p = &grad_alpha_map[map_index_from];


                    for (int d1 = d1_from; d1 <= d1_to; d1++) {
                        scalar_t diff_grad = 0;

                        diff_grad += (*alpha_map_p - alpha_in) * *grad_alpha_map_p;

                        alpha_map_p += map_offset;
                        grad_alpha_map_p += map_offset;

                        if (diff_grad <= 0)
                            continue;
                        printf("%f, %f, %f, %d, %d, %f\n", *grad_alpha_map_p, *alpha_map_p, alpha_in, d0, d1, diff_grad);
                        if (p[1][0] != d0) {
                            scalar_t dist = (p[1][0] - p[0][0]) / (p[1][0] - d0) * (d1 - d1_cross) * 2. / is;
                            dist = (0 < dist) ? dist + eps : dist - eps;
                            grad_face[pi[0] * 2 + (1 - axis)] -= diff_grad / dist;
                        }
                        if (p[0][0] != d0) {
                            scalar_t dist = (p[1][0] - p[0][0]) / (d0 - p[0][0]) * (d1 - d1_cross) * 2. / is;
                            dist = (0 < dist) ? dist + eps : dist - eps;
                            grad_face[pi[1] * 2 + (1 - axis)] -= diff_grad / dist;
                        }
                    }
                }
                /* in */
                {
                    int d1_limit;
                    scalar_t d0_cross2;
                    if ((d0 - p[0][0]) * (d0 - p[2][0]) < 0) {
                        d0_cross2 = (p[2][1] - p[0][1]) / (p[2][0] - p[0][0]) * (d0 - p[0][0]) + p[0][1];
                    }
                    else {
                        d0_cross2 = (p[1][1] - p[2][1]) / (p[1][0] - p[2][0]) * (d0 - p[2][0]) + p[2][1];
                    }
                    if (0 < direction)
                        d1_limit = ceil(d0_cross2);
                    else
                        d1_limit = floor(d0_cross2);
                    int d1_from = max(min(d1_in, d1_limit), 0);
                    int d1_to = min(max(d1_in, d1_limit), is - 1);

                    int* face_index_map_p;
                    scalar_t* alpha_map_p;
                    scalar_t* grad_alpha_map_p;
                    int map_index_from;
                    int map_offset;
                    if (axis == 0)
                        map_offset = is;
                    else
                        map_offset = 1;
                    if (axis == 0) {
                        map_index_from = bn * is * is + d1_from * is + d0;
                    }
                    else {
                        map_index_from = bn * is * is + d0 * is + d1_from;
                    }
                    face_index_map_p = &face_index_map[map_index_from] - map_offset;

                    alpha_map_p = &alpha_map[map_index_from] - map_offset;
                    grad_alpha_map_p = &grad_alpha_map[map_index_from] - map_offset;



                    for (int d1 = d1_from; d1 <= d1_to; d1++) {
                        face_index_map_p += map_offset;

                        alpha_map_p += map_offset;
                        grad_alpha_map_p += map_offset;

                        if (*face_index_map_p != fn)
                            continue;

                        scalar_t diff_grad = 0;
                        diff_grad += (*alpha_map_p - alpha_out) * *grad_alpha_map_p;

                        if (diff_grad <= 0)
                            continue;

                        if (p[1][0] != d0) {
                            scalar_t dist = (p[1][0] - p[0][0]) / (p[1][0] - d0) * (d1 - d1_cross) * 2. / is;
                            dist = (0 < dist) ? dist + eps : dist - eps;
                            grad_face[pi[0] * 2 + (1 - axis)] -= diff_grad / dist;
                        }
                        if (p[0][0] != d0) {
                            scalar_t dist = (p[1][0] - p[0][0]) / (d0 - p[0][0]) * (d1 - d1_cross) * 2. / is;
                            dist = (0 < dist) ? dist + eps : dist - eps;
                            grad_face[pi[1] * 2 + (1 - axis)] -= diff_grad / dist;
                        }
                    }
                }
            }
        }
    }

    /* set to global gradient variable */
    for (int k = 0; k < 6; k++)
        grad_faces[i * 6 + k] = grad_face[k];
}



at::Tensor forward_face_index_map_cuda(
        at::Tensor faces,
        at::Tensor face_index_map,
        int image_size) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const int threads = 512;
    const dim3 blocks_1 ((batch_size * num_faces - 1) / threads +1);


    const dim3 blocks_2 ((batch_size * image_size * image_size - 1) / threads +1);
    AT_DISPATCH_FLOATING_TYPES(faces.type(), "forward_face_index_map_cuda_2", ([&] {
      forward_face_index_map_cuda_kernel_2<scalar_t><<<blocks_2, threads>>>(
          faces.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          (int) batch_size,
          (int) num_faces,
          (int) image_size);
      }));
    cudaError_t err = cudaGetLastError();
    err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in forward_face_index_map_2: %s\n", cudaGetErrorString(err));
    return face_index_map;
}


at::Tensor backward_pixel_map_cuda(
        at::Tensor faces,
        at::Tensor face_index_map,
        at::Tensor alpha_map,
        at::Tensor grad_alpha_map,
        at::Tensor grad_faces,
        int image_size,
        float eps) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const int threads = 512;
    const dim3 blocks ((batch_size * num_faces - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "backward_pixel_map_cuda", ([&] {
      backward_pixel_map_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          face_index_map.data<int32_t>(),
          alpha_map.data<scalar_t>(),
          grad_alpha_map.data<scalar_t>(),
          grad_faces.data<scalar_t>(),
          batch_size,
		  num_faces,
          image_size,
          (scalar_t) eps);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in backward_pixel_map: %s\n", cudaGetErrorString(err));

    return grad_faces;
}

