// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GetParticlesHAbove.cpp
 * @author Jacksen Narvaez (jjnarvaezcor@mun.ca)
 * @brief Implements the GetParticlesHAbove module, which selects the particles whose smoothing
 * length is above a threshold.
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shammodels/sph/modules/GetParticlesHAbove.hpp"
#include "shamrock/patch/PatchDataField.hpp"

namespace shammodels::sph::modules {

    template<typename Tvec>
    void GetParticlesHAbove<Tvec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        const shamrock::solvergraph::DDPatchDataFieldRef<Tscal> &hpart_refs
            = edges.hpart.get_refs();

        edges.part_ids_h_above.ensure_allocated(hpart_refs.get_ids());

        hpart_refs.for_each([&](u64 id_patch, const PatchDataField<Tscal> &hpart) {
            auto tmp = hpart.get_ids_where(
                [](const Tscal *__restrict hpart, u32 i, Tscal h_max) {
                    return hpart[i] > h_max;
                },
                h_max);

            edges.part_ids_h_above.buffers.get(id_patch).append(tmp);
        });
    }

    template<typename Tvec>
    std::string GetParticlesHAbove<Tvec>::_impl_get_tex() const {
        auto hpart            = get_ro_edge_base(0).get_tex_symbol();
        auto part_ids_h_above = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
        Get particles with a smoothing length above a threshold

        \begin{align}
        {part_ids_h_above} &= \{i \text{ where } {hpart}_i > h_{\rm max}\}\\
        h_{\rm max} &= {h_max}
        \end{align}
        )tex";

        shambase::replace_all(tex, "{hpart}", hpart);
        shambase::replace_all(tex, "{part_ids_h_above}", part_ids_h_above);
        shambase::replace_all(tex, "{h_max}", sham::format("{}", h_max));

        return tex;
    }

} // namespace shammodels::sph::modules

template class shammodels::sph::modules::GetParticlesHAbove<f64_3>;
