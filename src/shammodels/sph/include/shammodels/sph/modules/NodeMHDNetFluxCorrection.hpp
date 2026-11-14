// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file NodeMHDNetFluxCorrection.hpp
 * @author Jacksen Narvaez (jjnarvaezcor@mun.ca)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsys/NodeInstance.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
                                                                                                   \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, gpart_mass)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, hfactd)                                          \
    X_RO(shamrock::solvergraph::IDataEdge<Tvec>, shear_grad)                                       \
    X_RO(shamrock::solvergraph::IDataEdge<Tvec>, shear_dir)                                        \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldRefs<Tscal>, hpart)                                          \
    X_RO(shamrock::solvergraph::IFieldRefs<Tvec>, B_on_rho)                                        \
                                                                                                   \
    /* inout */                                                                                    \
    X_RW(shamrock::solvergraph::IFieldRefs<Tvec>, dB_on_rho)

namespace shammodels::sph::modules {

    /**
     * @brief Remove the spurious change of the net magnetic flux from d(B/rho)/dt.
     *
     * The SPH induction equation does not conserve the net flux
     * \f$ \sum_a m_a (\mathbf{B}/\rho)_a \f$ of a periodic box. This node subtracts a uniform
     * \f$ d\mathbf{B}/dt \f$ such that the net flux only changes through the shear of the
     * boundaries (no change at all for periodic boundaries).
     */
    template<class Tvec>
    class NodeMHDNetFluxCorrection : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        NodeMHDNetFluxCorrection() {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
            sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

            Tscal pmass  = edges.gpart_mass.data;
            Tscal hfactd = edges.hfactd.data;

            Tvec sum_dB_on_rho{0, 0, 0};
            Tvec sum_B_on_rho{0, 0, 0};
            Tscal sum_inv_rho = 0;

            edges.part_counts.indexes.for_each_ordered([&](u64 id_patch, const u32 &cnt) {
                if (cnt == 0) {
                    return;
                }

                sham::DeviceBuffer<Tscal> inv_rho(cnt, dev_sched);

                sham::kernel_call(
                    q,
                    sham::MultiRef{edges.hpart.get_field(id_patch).get_buf()},
                    sham::MultiRef{inv_rho},
                    cnt,
                    [pmass, hfactd](u32 id_a, const Tscal *hpart, Tscal *inv_rho) {
                        inv_rho[id_a] = 1 / shamrock::sph::rho_h(pmass, hpart[id_a], hfactd);
                    });

                sum_dB_on_rho += shamalgs::primitives::sum(
                    dev_sched, edges.dB_on_rho.get_field(id_patch).get_buf(), 0, cnt);
                sum_B_on_rho += shamalgs::primitives::sum(
                    dev_sched, edges.B_on_rho.get_field(id_patch).get_buf(), 0, cnt);
                sum_inv_rho += shamalgs::primitives::sum(dev_sched, inv_rho, 0, cnt);
            });

            sum_dB_on_rho = shamalgs::collective::allreduce_sum(sum_dB_on_rho);
            sum_B_on_rho  = shamalgs::collective::allreduce_sum(sum_B_on_rho);
            sum_inv_rho   = shamalgs::collective::allreduce_sum(sum_inv_rho);

            // the shear of the boundaries is the only physical source of net flux
            Tvec sum_phys = sycl::dot(sum_B_on_rho, edges.shear_grad.data) * edges.shear_dir.data;
            Tvec corr     = (sum_dB_on_rho - sum_phys) / sum_inv_rho;

            sham::distributed_data_kernel_call(
                dev_sched,
                sham::DDMultiRef{edges.hpart.get_spans()},
                sham::DDMultiRef{edges.dB_on_rho.get_spans()},
                edges.part_counts.indexes,
                [pmass, hfactd, corr](
                    u32 id_a, const Tscal *__restrict hpart, Tvec *__restrict dB_on_rho) {
                    // corr / rho_a changes B_a by the same amount for every particle
                    dB_on_rho[id_a] -= corr / shamrock::sph::rho_h(pmass, hpart[id_a], hfactd);
                });
        }

        inline virtual std::string _impl_get_label() const { return "NodeMHDNetFluxCorrection"; };

        inline virtual std::string _impl_get_tex() const {
            std::string tex = R"tex(
                NodeMHDNetFluxCorrection

                With $\rho_a = \rho({hpart}_a)$, $m = {gpart_mass}$ and
                $h_{{\rm fact}} = {hfactd}$:

                \begin{align}
                \mathbf{c} &= \frac{\sum_b \left({dB_on_rho}\right)_b
                    - \left[\sum_b \left({B_on_rho}\right)_b \cdot {shear_grad}\right] {shear_dir}}
                    {\sum_b 1/\rho_b} \\
                \left({dB_on_rho}\right)_a &\leftarrow
                    \left({dB_on_rho}\right)_a - \frac{\mathbf{c}}{\rho_a}
                \end{align}

                $a, b \in [0,{part_counts})$.
            )tex";

            replace_edges_tex_symbols(tex);

            return tex;
        }
    };
} // namespace shammodels::sph::modules

#undef NODE_EDGES
