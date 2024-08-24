#include <pybind11/include/pybind11/pybind11.h>
#include <pybind11/include/pybind11/numpy.h>
#include <pybind11/include/pybind11/stl.h>
git 
namespace py = pybind11;

// Convert Image structure to use NumPy arrays
struct Image {
    py::array_t<unsigned char> data;
    int w;
    int h;
    int elempack;

    Image(py::array_t<unsigned char> d, int width, int height, int channels)
        : data(d), w(width), h(height), elempack(channels) {}
};

PYBIND11_MODULE(rife_module, m) {
    py::class_<Image>(m, "Image")
        .def(py::init<py::array_t<unsigned char>, int, int, int>())
        .def_readwrite("data", &Image::data)
        .def_readwrite("w", &Image::w)
        .def_readwrite("h", &Image::h)
        .def_readwrite("elempack", &Image::elempack);

    py::class_<RifeWrapped, RIFE>(m, "RifeWrapped")
        .def(py::init<int, bool, bool, bool, int, bool, bool>(),
             py::arg("gpuid"), py::arg("tta_mode"), py::arg("tta_temporal_mode"),
             py::arg("uhd_mode"), py::arg("num_threads"), py::arg("rife_v2"),
             py::arg("rife_v4"))
        .def("load", &RifeWrapped::load)
        .def("process", &RifeWrapped::process);

    m.def("get_gpu_count", &get_gpu_count, "Get the number of available GPUs");
}