#include "rife_wrapped.h"
#include <pybind11/include/pybind11/numpy.h>

// Constructor implementation
RifeWrapped::RifeWrapped(int gpuid, bool _tta_mode, bool _tta_temporal_mode, bool _uhd_mode,
                         int _num_threads, bool _rife_v2, bool _rife_v4)
    : RIFE(gpuid, _tta_mode, _tta_temporal_mode, _uhd_mode,
           _num_threads, _rife_v2, _rife_v4)
{
}

// Load method implementation
int RifeWrapped::load(const StringType &modeldir)
{
#if _WIN32
    return RIFE::load(*modeldir.wstr);
#else
    return RIFE::load(*modeldir.str);
#endif
}

// Process method implementation that uses NumPy arrays
int RifeWrapped::process(py::array_t<unsigned char> inimage0,
                         py::array_t<unsigned char> inimage1,
                         float timestep,
                         py::array_t<unsigned char> outimage)
{
    // Extract buffer information from NumPy arrays
    py::buffer_info buf0 = inimage0.request();
    py::buffer_info buf1 = inimage1.request();
    py::buffer_info buf_out = outimage.request();

    int c = buf0.shape[2]; // Assuming the last dimension is the channel count

    // Convert NumPy arrays to ncnn::Mat
    ncnn::Mat inimagemat0 = ncnn::Mat(buf0.shape[1], buf0.shape[0], buf0.ptr, (size_t)c, c);
    ncnn::Mat inimagemat1 = ncnn::Mat(buf1.shape[1], buf1.shape[0], buf1.ptr, (size_t)c, c);
    ncnn::Mat outimagemat = ncnn::Mat(buf_out.shape[1], buf_out.shape[0], buf_out.ptr, (size_t)c, c);

    // Call the original RIFE process method
    return RIFE::process(inimagemat0, inimagemat1, timestep, outimagemat);
}

// Function to get GPU count
int get_gpu_count() { return ncnn::get_gpu_count(); }