// IQ3S: the gate/up launches of routed format IQ4_XS (experts_gu.cuh), the IQ3_S file's layer 47, one file a routed
// format so the instantiations compile in parallel. IQ4_XS is an IQ format (kvalues_iq4nl, 136 B a 256), not a K-quant:
// it runs the deployed per-triple kernels like the IQ2 and IQ3 formats, not the K-quant family body.
#include "fn/kernels/experts/experts_gu.cuh"

namespace fn::experts::detail {

bool gate_up_iq4_xs(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st) {
    return gate_up_for<fmt::iq4_xs>(fg, fu, rows, a, st);
}

} // namespace fn::experts::detail
