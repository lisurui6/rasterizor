import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
import rasterizor.cuda.rasterize as rasterize_cuda
import neural_renderer as nr

DEFAULT_IMAGE_SIZE = 256
DEFAULT_ANTI_ALIASING = True
DEFAULT_EPS = 1e-4
DEFAULT_BACKGROUND_COLOR = (0, 0, 0)


class TriangleRasterizeFunction(Function):
    '''
    Definition of differentiable rasterize operation
    Some parts of the code are implemented in CUDA
    Currently implemented only for cuda Tensors
    '''

    @staticmethod
    def forward(ctx, faces, image_size):
        ctx.image_size = image_size
        faces = faces.clone()
        ctx.device = faces.device
        ctx.batch_size, ctx.num_faces = faces.shape[:2]

        # initializing with dummy values
        face_index_map = torch.cuda.IntTensor(ctx.batch_size, ctx.image_size, ctx.image_size).fill_(-1)
        alpha_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_size, ctx.image_size).fill_(0)
        face_index_map = TriangleRasterizeFunction.forward_face_index_map(ctx, faces, face_index_map)
        alpha_map[face_index_map >= 0] = 1
        ctx.save_for_backward(faces, face_index_map, alpha_map)
        alpha_r = alpha_map.clone()
        return alpha_r

    @staticmethod
    def backward(ctx, grad_alpha_map):
        faces, face_index_map, alpha_map = ctx.saved_tensors
        # initialize output buffers
        # no need for explicit allocation of cuda.FloatTensor because zeros_like does it automatically
        grad_faces = torch.zeros_like(faces, dtype=torch.float32)

        # get grad_outputs

        if grad_alpha_map is not None:
            grad_alpha_map = grad_alpha_map.contiguous()
        else:
            grad_alpha_map = torch.zeros_like(alpha_map)

        # backward pass
        grad_faces = TriangleRasterizeFunction.backward_pixel_map(
            ctx, faces, face_index_map,
            alpha_map, grad_alpha_map,
            grad_faces)

        return grad_faces, None

    @staticmethod
    def forward_face_index_map(ctx, faces, face_index_map):
        return rasterize_cuda.forward_face_index_map(faces, face_index_map, ctx.image_size)

    @staticmethod
    def backward_pixel_map(ctx, faces, face_index_map,
                           alpha_map, grad_alpha_map, grad_faces):

        return rasterize_cuda.backward_pixel_map(faces, face_index_map,
                                                 alpha_map, grad_alpha_map,
                                                 grad_faces, ctx.image_size, DEFAULT_EPS)


class Rasterize(nn.Module):
    '''
    Wrapper around the autograd function RasterizeFunction
    Currently implemented only for cuda Tensors
    '''

    def __init__(self, image_size, anti_aliasing=True, mode="triangle"):
        super(Rasterize, self).__init__()
        if anti_aliasing:
            image_size = image_size * 2
        self.anti_aliasing = anti_aliasing
        self.image_size = image_size
        self.mode = mode

    @staticmethod
    def vertices_to_faces(vertices, faces):
        bs, nv = vertices.shape[:2]
        bs, nf = faces.shape[:2]
        device = vertices.device
        faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        vertices = vertices.reshape((bs * nv, 2))
        return vertices[faces.long()]

    def forward(self, vertices, faces):
        """
        :param vertices: (B, N_v, 2)
        :param faces: (B, N_f, 3)
        :return: (B, image_size, image_size)
        """
        if faces.device == "cpu":
            raise TypeError('Rasterize module supports only cuda Tensors')
        # fill back
        faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1)
        # rasterization
        faces = self.vertices_to_faces(vertices, faces)

        if self.mode == "triangle":
            if self.anti_aliasing:
                alpha = TriangleRasterizeFunction.apply(faces, self.image_size * 2)
            else:
                alpha = TriangleRasterizeFunction.apply(faces, self.image_size)

        # transpose & vertical flip
        alpha = alpha[:, list(reversed(range(alpha.shape[1]))), :]
        if self.anti_aliasing:
            # 0.5x down-sampling
            alpha = F.avg_pool2d(alpha[:, None, :, :], kernel_size=(2, 2))[:, 0]
        return alpha
