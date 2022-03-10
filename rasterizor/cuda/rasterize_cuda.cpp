#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor forward_face_index_map_cuda(
        at::Tensor faces,
        at::Tensor face_index_map,
        int image_size);

at::Tensor backward_pixel_map_cuda(
        at::Tensor faces,
        at::Tensor face_index_map,
        at::Tensor alpha_map,
        at::Tensor grad_alpha_map,
        at::Tensor grad_faces,
        int image_size,
        float eps);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor forward_face_index_map(
        at::Tensor faces,
        at::Tensor face_index_map,
        int image_size) {

    CHECK_INPUT(faces);
    CHECK_INPUT(face_index_map);

    return forward_face_index_map_cuda(faces, face_index_map,
                                       image_size);
}


at::Tensor backward_pixel_map(
        at::Tensor faces,
        at::Tensor face_index_map,
        at::Tensor alpha_map,
        at::Tensor grad_alpha_map,
        at::Tensor grad_faces,
        int image_size,
        float eps) {

    CHECK_INPUT(faces);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(alpha_map);
    CHECK_INPUT(grad_alpha_map);
    CHECK_INPUT(grad_faces);

    return backward_pixel_map_cuda(faces, face_index_map, alpha_map,
                                   grad_alpha_map, grad_faces,
                                   image_size, eps);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_face_index_map", &forward_face_index_map, "FORWARD_FACE_INDEX_MAP (CUDA)");
    m.def("backward_pixel_map", &backward_pixel_map, "BACKWARD_PIXEL_MAP (CUDA)");
}
