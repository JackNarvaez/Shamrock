// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeDivBProjectionPrecond.cpp
 * @author Jacksen Narvaez (jjnarvaezcor@mun.ca)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/NodeDivBProjectionPrecond.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::NodeDivBProjectionPrecond<Tvec, SPHKernel>::
    _impl_evaluate_internal() {

    __shamrock_stack_entry();
    auto edges = get_edges();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
    edges.inv_diag.ensure_sizes(edges.part_counts.indexes);

    sham::distributed_data_kernel_call(
        dev_sched,
        sham::DDMultiRef{
            edges.xyz.get_spans(),
            edges.hpart.get_spans(),
            edges.neigh_cache.neigh_cache,
            edges.omega.get_spans()},
        sham::DDMultiRef{edges.inv_diag.get_spans()},
        edges.part_counts.indexes,
        [part_mass = this->part_mass](
            u32 id_a,
            const Tvec *r,
            const Tscal *hpart,
            const auto ploop_ptrs,
            const Tscal *omega,
            Tscal *inv_diag) {
            shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            using namespace shamrock::sph;

            Tvec xyz_a = r[id_a];

            Tscal h_a = hpart[id_a];

            Tscal rho_a      = rho_h(part_mass, h_a, SPHKernel<Tscal>::hfactd);
            Tscal omega_a    = omega[id_a];
            Tscal sub_fact_a = rho_a * omega_a;

            Tvec d_ab_sum{0, 0, 0};
            Tscal d_ab_sq_sum = 0;

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

                Tscal Fab_a    = SPHKernel<Tscal>::dW_3d(rab, h_a);
                Tvec r_ab_unit = dr * sham::inv_sat_positive(rab);

                Tvec d_ab = r_ab_unit * (part_mass * Fab_a * sham::inv_sat_zero(sub_fact_a));

                d_ab_sum += d_ab;
                d_ab_sq_sum += sycl::dot(d_ab, d_ab) * rho_b / part_mass;
            });

            Tscal diag = sycl::dot(d_ab_sum, d_ab_sum) * rho_a / part_mass + d_ab_sq_sum;

            inv_diag[id_a] = sham::inv_sat_zero(diag);
        });
}

template<class Tvec, template<class> class SPHKernel>
std::string shammodels::sph::modules::NodeDivBProjectionPrecond<Tvec, SPHKernel>::_impl_get_tex()
    const {
    return "TODO";
}

using namespace shammath;
template class shammodels::sph::modules::NodeDivBProjectionPrecond<f64_3, M4>;
template class shammodels::sph::modules::NodeDivBProjectionPrecond<f64_3, M6>;
template class shammodels::sph::modules::NodeDivBProjectionPrecond<f64_3, M8>;

template class shammodels::sph::modules::NodeDivBProjectionPrecond<f64_3, C2>;
template class shammodels::sph::modules::NodeDivBProjectionPrecond<f64_3, C4>;
template class shammodels::sph::modules::NodeDivBProjectionPrecond<f64_3, C6>;
