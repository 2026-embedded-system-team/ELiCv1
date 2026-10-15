#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "arithmetic_core.h"

namespace py = pybind11;

static void check_cdf_16(const torch::Tensor& cdf) {
    TORCH_CHECK(cdf.device().is_cpu(), "cdf must be on CPU");
    TORCH_CHECK(cdf.dim() == 2 && cdf.size(1) == 17, "cdf must have shape (N,17)");
    TORCH_CHECK(cdf.scalar_type() == torch::kInt16, "cdf must be torch.int16 (kShort)");
    TORCH_CHECK(cdf.is_contiguous(), "cdf must be contiguous");
}

class ArithmeticEncoder16 {
public:
    void reset(bool reset_output = true) { core_.reset(reset_output); }
    void encode_chunk(const torch::Tensor& cdf, const torch::Tensor& sym) {
        check_cdf_16(cdf);
        TORCH_CHECK(sym.device().is_cpu(), "sym must be on CPU");
        TORCH_CHECK(sym.dim() == 1, "sym must be 1D (N)");
        TORCH_CHECK(sym.scalar_type() == torch::kInt16, "sym must be torch.int16 (kShort)");
        TORCH_CHECK(sym.is_contiguous(), "sym must be contiguous");
        TORCH_CHECK(sym.size(0) == cdf.size(0), "sym length must match cdf.size(0)");
        core_.encode_chunk(reinterpret_cast<const uint16_t*>(cdf.data_ptr<int16_t>()),
                           sym.data_ptr<int16_t>(), cdf.size(0));
    }
    py::bytes finish(bool reset_after = true, bool clear_output_after = true) {
        core_.finalize();
        py::bytes result(core_.output.out);
        if (reset_after) core_.reset(clear_output_after);
        return result;
    }
    py::bytes get_buffer_bytes() const { return py::bytes(core_.output.out); }
    int buffered_bitcount() const { return static_cast<int>(core_.output.count); }
private:
    elic::ArithmeticEncoder16 core_;
};

class ArithmeticDecoder16 {
public:
    void reset(const py::bytes& bitstream) { core_.reset(std::string(bitstream)); }
    torch::Tensor decode_chunk(const torch::Tensor& cdf) {
        check_cdf_16(cdf);
        auto out = torch::empty({cdf.size(0)}, torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU));
        core_.decode_chunk(reinterpret_cast<const uint16_t*>(cdf.data_ptr<int16_t>()),
                           out.data_ptr<int16_t>(), cdf.size(0));
        return out;
    }
    int64_t bytes_consumed() const { return core_.bytes_consumed(); }
    int64_t bytes_remaining() const { return core_.bytes_remaining(); }
private:
    elic::ArithmeticDecoder16 core_;
};

static py::bytes encode_cdf_16(const torch::Tensor& cdf, const torch::Tensor& sym) {
    ArithmeticEncoder16 encoder;
    { py::gil_scoped_release release; encoder.encode_chunk(cdf, sym); }
    return encoder.finish(false, false);
}
static torch::Tensor decode_cdf_16(const torch::Tensor& cdf, const py::bytes& bitstream) {
    ArithmeticDecoder16 decoder;
    decoder.reset(bitstream);
    py::gil_scoped_release release;
    return decoder.decode_chunk(cdf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_cdf_16", &encode_cdf_16, "Encode single chunk from 16-way CDF (N,17)");
    m.def("decode_cdf_16", &decode_cdf_16, "Decode single chunk from 16-way CDF (N,17)");
    py::class_<ArithmeticEncoder16>(m, "ArithmeticEncoder16")
        .def(py::init<>())
        .def("reset", &ArithmeticEncoder16::reset, py::arg("reset_output") = true)
        .def("encode_chunk", &ArithmeticEncoder16::encode_chunk, py::arg("cdf"), py::arg("sym"),
             py::call_guard<py::gil_scoped_release>())
        .def("finish", &ArithmeticEncoder16::finish,
             py::arg("reset_after") = true, py::arg("clear_output_after") = true)
        .def("get_buffer_bytes", &ArithmeticEncoder16::get_buffer_bytes)
        .def("buffered_bitcount", &ArithmeticEncoder16::buffered_bitcount);
    py::class_<ArithmeticDecoder16>(m, "ArithmeticDecoder16")
        .def(py::init<>())
        .def("reset", &ArithmeticDecoder16::reset, py::arg("bitstream"))
        .def("decode_chunk", &ArithmeticDecoder16::decode_chunk, py::arg("cdf"),
             py::call_guard<py::gil_scoped_release>())
        .def("bytes_consumed", &ArithmeticDecoder16::bytes_consumed)
        .def("bytes_remaining", &ArithmeticDecoder16::bytes_remaining);
}
