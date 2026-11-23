// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeDivBDifference.cpp
 * @author Jacksen Narvaez (jjnarvaezcor@mun.ca)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/math/mhd.hpp"
#include "shammodels/sph/modules/NodeDivBDifference.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::NodeDivBDifference<Tvec, SPHKernel>::_impl_evaluate_internal() {

    __shamrock_stack_entry();
    auto edges = get_edges();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    edges.div_B.ensure_sizes(edges.part_counts.indexes);

    sham::distributed_data_kernel_call(
        dev_sched,
        sham::DDMultiRef{
            edges.xyz.get_spans(),
            edges.hpart.get_spans(),
            edges.neigh_cache.neigh_cache,
            edges.omega.get_spans(),
            edges.B_on_rho.get_spans()},
        sham::DDMultiRef{edges.div_B.get_spans()},
        edges.part_counts.indexes,
        [part_mass = this->part_mass](
            u32 id_a,
            const Tvec *r,
            const Tscal *hpart,
            const auto ploop_ptrs,
            const Tscal *omega,
            const Tvec *B_on_rho,
            Tscal *div_B) {
            shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            using namespace shamrock::sph;
            using namespace shamrock::sph::mhd;

            Tvec xyz_a = r[id_a];

            Tscal h_a = hpart[id_a];

            Tscal rho_a = rho_h(part_mass, h_a, SPHKernel<Tscal>::hfactd);

            Tvec B_a      = B_on_rho[id_a] * rho_a;
            Tscal omega_a = omega[id_a];

            Tscal div_B_sum = 0;

            constexpr Tscal rker2 = SPHKernel<Tscal>::Rkern * SPHKernel<Tscal>::Rkern;

            particle_looper.for_each_object(id_a, [&](u32 id_b) {
                Tvec dr    = xyz_a - r[id_b];
                Tscal rab2 = sycl::dot(dr, dr);
                Tscal h_b  = hpart[id_b];

                if (rab2 > h_a * h_a * rker2 && rab2 > h_b * h_b * rker2) {
                    return;
                }

                Tscal rab   = sycl::sqrt(rab2);
                Tscal rho_b = rho_h(part_mass, h_b, SPHKernel<Tscal>::hfactd);
                Tvec B_b    = B_on_rho[id_b] * rho_b;

                Tscal Fab_a       = SPHKernel<Tscal>::dW_3d(rab, h_a);
                Tvec r_ab_unit    = dr * sham::inv_sat_positive(rab);
                Tvec nabla_Wab_ha = r_ab_unit * Fab_a;

                div_B_sum += shamrock::sph::mhd::dpsi_on_ch_parabolic_propag(
                    part_mass, rho_a, B_a, B_b, omega_a, nabla_Wab_ha, Tscal(-1));
            });

            div_B[id_a] = div_B_sum;
        });
}

template<class Tvec, template<class> class SPHKernel>
std::string shammodels::sph::modules::NodeDivBDifference<Tvec, SPHKernel>::_impl_get_tex() const {
    return "TODO";
}

using namespace shammath;
template class shammodels::sph::modules::NodeDivBDifference<f64_3, M4>;
template class shammodels::sph::modules::NodeDivBDifference<f64_3, M6>;
template class shammodels::sph::modules::NodeDivBDifference<f64_3, M8>;

template class shammodels::sph::modules::NodeDivBDifference<f64_3, C2>;
template class shammodels::sph::modules::NodeDivBDifference<f64_3, C4>;
template class shammodels::sph::modules::NodeDivBDifference<f64_3, C6>;
