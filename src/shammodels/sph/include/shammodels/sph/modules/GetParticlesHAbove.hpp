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
 * @file GetParticlesHAbove.hpp
 * @author Jacksen Narvaez (jjnarvaezcor@mun.ca)
 * @brief Declares the GetParticlesHAbove module, which selects the particles whose smoothing
 * length is above a threshold.
 *
 */

#include "shamrock/solvergraph/DistributedBuffers.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamsolvergraph/node/INode.hpp"

namespace shammodels::sph::modules {

    template<typename Tvec>
    class GetParticlesHAbove : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        Tscal h_max;

        public:
        GetParticlesHAbove(Tscal h_max) : h_max(h_max) {}

        struct Edges {
            const shamrock::solvergraph::IFieldRefs<Tscal> &hpart;
            shamrock::solvergraph::DistributedBuffers<u32> &part_ids_h_above;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tscal>> hpart,
            std::shared_ptr<shamrock::solvergraph::DistributedBuffers<u32>> part_ids_h_above) {
            __internal_set_ro_edges({hpart});
            __internal_set_rw_edges({part_ids_h_above});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(0),
                get_rw_edge<shamrock::solvergraph::DistributedBuffers<u32>>(0)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "GetParticlesHAbove"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::sph::modules
