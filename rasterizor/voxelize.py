import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
import rasterizor.cuda.voxelize as voxelize_cuda
import neural_renderer as nr

DEFAULT_IMAGE_SIZE = 256
DEFAULT_ANTI_ALIASING = True
DEFAULT_EPS = 1e-4
DEFAULT_BACKGROUND_COLOR = (0, 0, 0)




class VoxelizeFunction(Function):
    """
    Definition of differentiable voxelize operation
    Some parts of the code are implemented in CUDA
    Currently implemented only for cuda Tensors
    """

    @staticmethod
    def forward(ctx, facets, voxel_width, voxel_depth, voxel_height):
        ctx.voxel_size = (voxel_width, voxel_depth, voxel_height)
        facets = facets.clone()
        ctx.device = facets.device
        ctx.batch_size, ctx.num_faces = facets.shape[:2]

        # initializing with dummy values
        facet_index_map = torch.cuda.IntTensor(ctx.batch_size, voxel_width, voxel_depth, voxel_height).fill_(-1)
        alpha_map = torch.cuda.FloatTensor(ctx.batch_size, voxel_width, voxel_depth, voxel_height).fill_(0)

        for b in range(ctx.batch_size):
            for xi in range(voxel_width):
                xp = (2. * xi + 1 - voxel_width) / voxel_width
                for yi in range(voxel_depth):
                    yp = (2. * yi + 1 - voxel_depth) / voxel_depth
                    for zi in range(voxel_height):
                        zp = (2. * zi + 1 - voxel_height) / voxel_height
                        for facet_num in range(ctx.num_faces):
                            # t11 = facet[0] - facet[9]
                            t11 = facets[b, facet_num, 0, 0] - facets[b, facet_num, 3, 0]
                            # t12 = facet[3] - facet[9]
                            t12 = facets[b, facet_num, 1, 0] - facets[b, facet_num, 3, 0]
                            # t13 = facet[6] - facet[9]
                            t13 = facets[b, facet_num, 2, 0] - facets[b, facet_num, 3, 0]
                            # t21 = facet[1] - facet[10]
                            t21 = facets[b, facet_num, 0, 1] - facets[b, facet_num, 3, 1]
                            # t22 = facet[4] - facet[10]
                            t22 = facets[b, facet_num, 1, 1] - facets[b, facet_num, 3, 1]
                            # t23 = facet[7] - facet[10]
                            t23 = facets[b, facet_num, 2, 1] - facets[b, facet_num, 3, 1]
                            # t31 = facet[2] - facet[11]
                            t31 = facets[b, facet_num, 0, 2] - facets[b, facet_num, 3, 2]
                            # t32 = facet[5] - facet[11]
                            t32 = facets[b, facet_num, 1, 2] - facets[b, facet_num, 3, 2]
                            # t33 = facet[8] - facet[11]
                            t33 = facets[b, facet_num, 2, 2] - facets[b, facet_num, 3, 2]

                            t_inv_11 = t22 * t33 - t23 * t32
                            t_inv_12 = t13 * t32 - t12 * t33
                            t_inv_13 = t12 * t23 - t13 * t22
                            t_inv_21 = t23 * t31 - t21 * t33
                            t_inv_22 = t11 * t33 - t13 * t31
                            t_inv_23 = t13 * t21 - t11 * t23
                            t_inv_31 = t21 * t32 - t22 * t31
                            t_inv_32 = t12 * t31 - t11 * t32
                            t_inv_33 = t11 * t22 - t12 * t21
                            t_det = t11 * (t22 * t33 - t23 * t32) - t12 * (t21 * t33 - t23 * t31) + t13 * (
                                        t21 * t32 - t22 * t31)

                            lambda_1 = (t_inv_11 * (xp - facets[b, facet_num, 3, 0]) + t_inv_12 * (yp - facets[b, facet_num, 3, 1]) + t_inv_13 * (
                                        zp - facets[b, facet_num, 3, 2])) / t_det
                            lambda_2 = (t_inv_21 * (xp - facets[b, facet_num, 3, 0]) + t_inv_22 * (yp - facets[b, facet_num, 3, 1]) + t_inv_23 * (
                                        zp - facets[b, facet_num, 3, 2])) / t_det
                            lambda_3 = (t_inv_31 * (xp - facets[b, facet_num, 3, 0]) + t_inv_32 * (yp - facets[b, facet_num, 3, 1]) + t_inv_33 * (
                                        zp - facets[b, facet_num, 3, 2])) / t_det
                            lambda_4 = 1 - lambda_1 - lambda_2 - lambda_3
                            if facet_num == 6:
                                print("here")
                            if ((lambda_1 < 0) or (lambda_2 < 0) or (lambda_3 < 0) or (lambda_4 < 0) or
                                    (lambda_1 > 1) or (lambda_2 > 1) or (lambda_3 > 1) or (lambda_4 > 1)):
                                continue
                            facet_index_min = facet_num
                            break

                        if 0 <= facet_index_min:
                            facet_index_map[b, xi, yi, zi] = facet_index_min

        alpha_map[facet_index_map >= 0] = 1
        ctx.save_for_backward(facets, facet_index_map, alpha_map)
        alpha_r = alpha_map.clone()
        return alpha_r


class VoxelizeCudaFunction(Function):
    """
    Definition of differentiable voxelize operation
    Some parts of the code are implemented in CUDA
    Currently implemented only for cuda Tensors
    """

    @staticmethod
    def forward(ctx, facets, voxel_width, voxel_depth, voxel_height, eps, eps_in):
        ctx.voxel_size = (voxel_width, voxel_depth, voxel_height)
        facets = facets.clone()
        ctx.device = facets.device
        ctx.batch_size, ctx.num_faces = facets.shape[:2]
        ctx.eps = eps
        ctx.eps_in = eps_in

        # initializing with dummy values
        facet_index_map = torch.cuda.IntTensor(ctx.batch_size, voxel_width, voxel_depth, voxel_height).fill_(-1)
        alpha_map = torch.cuda.FloatTensor(ctx.batch_size, voxel_width, voxel_depth, voxel_height).fill_(0)
        facet_index_map = VoxelizeCudaFunction.forward_facet_index_map(ctx, facets, facet_index_map)
        alpha_map[facet_index_map >= 0] = 1
        ctx.save_for_backward(facets, facet_index_map, alpha_map)
        alpha_r = alpha_map.clone()
        return alpha_r

    @staticmethod
    def backward(ctx, grad_alpha_map):
        facets, facet_index_map, alpha_map = ctx.saved_tensors
        # initialize output buffers
        # no need for explicit allocation of cuda.FloatTensor because zeros_like does it automatically
        grad_facets = torch.zeros_like(facets, dtype=torch.float32)

        # get grad_outputs

        if grad_alpha_map is not None:
            grad_alpha_map = grad_alpha_map.contiguous()
        else:
            grad_alpha_map = torch.zeros_like(alpha_map)

        # backward pass
        grad_facets = VoxelizeCudaFunction.backward_voxel_map(
            ctx, facets, facet_index_map,
            alpha_map, grad_alpha_map,
            grad_facets)

        return grad_facets, None, None, None, None, None

    @staticmethod
    def forward_facet_index_map(ctx, facets, facet_index_map):
        voxel_width, voxel_depth, voxel_height = ctx.voxel_size
        return voxelize_cuda.forward_facet_index_map(
            facets, facet_index_map, voxel_width, voxel_depth, voxel_height
        )

    @staticmethod
    def backward_voxel_map(ctx, facets, facet_index_map,
                           alpha_map, grad_alpha_map, grad_facets):
        voxel_width, voxel_depth, voxel_height = ctx.voxel_size
        return voxelize_cuda.backward_voxel_map(
            facets, facet_index_map, alpha_map, grad_alpha_map, grad_facets,
            voxel_width, voxel_depth, voxel_height, ctx.eps, ctx.eps_in
        )


class Voxelize(nn.Module):
    """
    Wrapper around the autograd function RasterizeFunction
    Currently implemented only for cuda Tensors
    """

    def __init__(self, voxel_width, voxel_depth, voxel_height, anti_aliasing=True, eps=DEFAULT_EPS, eps_in=0.5):
        super().__init__()

        self.anti_aliasing = anti_aliasing
        self.voxel_width = voxel_width
        self.voxel_depth = voxel_depth
        self.voxel_height = voxel_height
        self.eps = eps
        self.eps_in = eps_in

    @staticmethod
    def vertices_to_facets(vertices, facets):
        bs, nv = vertices.shape[:2]
        bs, nf = facets.shape[:2]
        device = vertices.device
        facets = facets + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        vertices = vertices.reshape((bs * nv, 3))
        return vertices[facets.long()]

    def forward(self, vertices, facets, use_cuda: bool = True):

        # fill back
        # facets = torch.cat((facets, facets[:, :, list(reversed(range(facets.shape[-1])))]), dim=1)

        # viewpoint transformation
        # vertices = nr.look_at(vertices, self.eye)
        # rasterization
        # facets = nr.vertices_to_faces(vertices, facets)
        facets = self.vertices_to_facets(vertices, facets)

        if not use_cuda:
            alpha = VoxelizeFunction.apply(facets, self.voxel_width, self.voxel_depth, self.voxel_height)
        else:
            alpha = VoxelizeCudaFunction.apply(
                facets, self.voxel_width, self.voxel_depth, self.voxel_height, self.eps, self.eps_in
            )
        # transpose & vertical flip
        # alpha = alpha[:, ::-1, :]
        # alpha = alpha[:, list(reversed(range(alpha.shape[1]))), :]
        # if self.anti_aliasing:
        #     # 0.5x down-sampling
        #     alpha = F.avg_pool2d(alpha[:, None, :, :], kernel_size=(2, 2))[:, 0]
        return alpha
