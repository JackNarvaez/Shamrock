// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeUpdateDerivsVaryingAlphaAV.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/modules/NodeUpdateDerivsVaryingAlphaAV_MHD.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/math/forces.hpp"
#include "shammodels/sph/math/mhd.hpp"
#include "shammodels/sph/math/q_ab.hpp"
#include "shamrock/patch/PatchDataField.hpp"

template<class Tvec, template<class> class SPHKernel>
struct KernelUpdateDerivsVaryingAlphaAV_MHD {
    using Tscal                   = shambase::VecComponent<Tvec>;
    using Kernel                  = SPHKernel<Tscal>;
    static constexpr Tscal hfactd = Kernel::hfactd;
    static constexpr Tscal Rkern  = Kernel::Rkern;
    static constexpr Tscal Rker2  = Rkern * Rkern;

    Tscal pmass;
    Tscal alpha_u;
    Tscal beta_AV;
    Tscal sigma_mhd;
    Tscal mu_0;

    inline void operator()(
        unsigned int id_a,
        const Tvec *__restrict xyz,
        const Tscal *__restrict hpart,
        const Tvec *__restrict vxyz,
        const Tscal *__restrict uint,
        const Tscal *__restrict omega,
        const Tscal *__restrict pressure,
        const Tscal *__restrict cs,
        const Tscal *__restrict alpha_AV,
        const Tvec *__restrict B_on_rho,
        const Tscal *__restrict psi_on_ch,
        shamrock::tree::ObjectCache::ptrs_read ploop_ptrs,
        Tvec *__restrict axyz,
        Tscal *__restrict duint,
        Tvec *__restrict dB_on_rho,
        Tscal *__restrict dpsi_on_ch,
        Tscal *__restrict drho_dt,
        Tscal *__restrict divB,
        Tvec *__restrict curlB,
        Tscal *__restrict dudtAV,
        Tscal *__restrict dudtAR) const {

        using namespace shamrock::sph;

        shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

        Tvec sum_axyz  = {0, 0, 0};
        Tscal sum_du_a = 0;

        Tvec xyz_a    = xyz[id_a];
        Tscal h_a     = hpart[id_a];
        Tvec vxyz_a   = vxyz[id_a];
        Tscal u_a     = uint[id_a];
        Tscal omega_a = omega[id_a];
        Tscal P_a     = pressure[id_a];
        Tscal cs_a    = cs[id_a];
        Tscal alpha_a = alpha_AV[id_a];

        Tscal rho_a     = rho_h(pmass, h_a, hfactd);
        Tscal rho_a_sq  = rho_a * rho_a;

        Tvec B_a         = B_on_rho[id_a] * rho_a;
        Tscal v_alfven_a = sycl::sqrt(sycl::dot(B_a, B_a) / (mu_0 * rho_a));
        Tscal v_shock_a  = sycl::sqrt(cs_a * cs_a + v_alfven_a * v_alfven_a);
        Tscal psi_a      = psi_on_ch[id_a] * v_shock_a;

        Tscal omega_a_rho_a_inv = 1 / (omega_a * rho_a);

        Tvec force_pressure  = Tvec{0, 0, 0};
        Tscal tmpdU_pressure = Tscal{0};
        Tvec magnetic_eq{0, 0, 0};
        Tscal psi_eq  = 0;
        Tscal drho_eq = 0;

        Tscal divB_a = 0;
        Tvec curlB_a{0, 0, 0};
        Tscal dudtAV_a = 0;
        Tscal dudtAR_a = 0;

        Tvec mag_pressure_term{0, 0, 0};
        Tvec mag_tension_term{0, 0, 0};
        Tvec gas_pressure_term{0, 0, 0};
        Tvec tensile_corr_term{0, 0, 0};

        Tscal psi_propag_term = 0;
        Tscal psi_diff_term   = 0;
        Tscal psi_cons_term   = 0;

        Tscal u_mhd_term = 0;

        particle_looper.for_each_object(id_a, [&](u32 id_b) {
            Tvec dr    = xyz_a - xyz[id_b];
            Tscal rab2 = sycl::dot(dr, dr);
            Tscal h_b  = hpart[id_b];

            if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                return;
            }

            Tvec vxyz_b         = vxyz[id_b];
            const Tscal u_b     = uint[id_b];
            Tscal P_b           = pressure[id_b];
            Tscal omega_b       = omega[id_b];
            const Tscal alpha_b = alpha_AV[id_b];
            Tscal cs_b          = cs[id_b];

            Tscal rab = sycl::sqrt(rab2);

            Tscal rho_b      = rho_h(pmass, h_b, hfactd);
            Tvec B_b         = B_on_rho[id_b] * rho_b;
            Tscal v_alfven_b = sycl::sqrt(sycl::dot(B_b, B_b) / (mu_0 * rho_b));
            Tscal v_shock_b  = sycl::sqrt(cs_b * cs_b + v_alfven_b * v_alfven_b);
            Tscal psi_b      = psi_on_ch[id_b] * v_shock_b;

            Tscal Fab_a = Kernel::dW_3d(rab, h_a);
            Tscal Fab_b = Kernel::dW_3d(rab, h_b);

            Tvec v_ab = vxyz_a - vxyz_b;

            Tvec r_ab_unit = dr * sham::inv_sat_positive(rab);

            Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);

            Tscal vsig_a = shamphys::MHD_physics<Tvec, Tscal>::vsig_MHD(
                v_ab, r_ab_unit, cs_a, B_a, rho_a, mu_0, alpha_a, beta_AV);
            Tscal vsig_b = shamphys::MHD_physics<Tvec, Tscal>::vsig_MHD(
                v_ab, r_ab_unit, cs_b, B_b, rho_b, mu_0, alpha_b, beta_AV);

            Tscal vsig_u = shamrock::sph::vsig_u(P_a, P_b, rho_a, rho_b);

            Tscal qa_ab = shamrock::sph::q_av(rho_a, vsig_a, v_ab_r_ab);
            Tscal qb_ab = shamrock::sph::q_av(rho_b, vsig_b, v_ab_r_ab);

            shamrock::sph::mhd::add_to_derivs_spmhd_av<Kernel, Tvec, Tscal>(
                pmass,
                rho_a_sq,
                omega_a_rho_a_inv,
                rho_a,
                rho_b,
                omega_a,
                omega_b,
                Fab_a,
                Fab_b,
                u_a,
                u_b,
                P_a,
                P_b,
                alpha_u,
                v_ab,
                r_ab_unit,
                vsig_u,
                qa_ab,
                qb_ab,

                cs_a,
                cs_b,

                B_a,
                B_b,

                psi_a,
                psi_b,

                mu_0,

                force_pressure,
                tmpdU_pressure,
                magnetic_eq,
                psi_eq,
                drho_eq,
                mag_pressure_term,
                mag_tension_term,
                gas_pressure_term,
                tensile_corr_term,

                psi_propag_term,
                psi_diff_term,
                psi_cons_term,
                u_mhd_term,
                divB_a,
                curlB_a,
                dudtAV_a,
                dudtAR_a);
        });

        axyz[id_a]       = force_pressure;
        duint[id_a]      = tmpdU_pressure;
        dB_on_rho[id_a]  = magnetic_eq;
        dpsi_on_ch[id_a] = psi_eq - sigma_mhd * psi_a / h_a;
        drho_dt[id_a]    = drho_eq;
        divB[id_a]       = divB_a;
        curlB[id_a]      = curlB_a;
        dudtAV[id_a]     = dudtAV_a;
        dudtAR[id_a]     = dudtAR_a;
    }
};

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::NodeUpdateDerivsVaryingAlphaAV_MHD<Tvec, SPHKernel>::
    _impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto &part_counts_with_ghost = edges.part_counts_with_ghost.indexes;
    auto &part_counts            = edges.part_counts.indexes;

    // check that all input edges have the particles with ghosts zones
    edges.xyz.check_sizes(part_counts_with_ghost);
    edges.hpart.check_sizes(part_counts_with_ghost);
    edges.vxyz.check_sizes(part_counts_with_ghost);
    edges.uint.check_sizes(part_counts_with_ghost);
    edges.omega.check_sizes(part_counts_with_ghost);
    edges.pressure.check_sizes(part_counts_with_ghost);
    edges.cs.check_sizes(part_counts_with_ghost);
    edges.alpha_AV.check_sizes(part_counts_with_ghost);
    edges.B_on_rho.check_sizes(part_counts_with_ghost);
    edges.psi_on_ch.check_sizes(part_counts_with_ghost);

    // ensure that the output edges are of size part_counts (output without ghosts zones)
    edges.axyz.ensure_sizes(part_counts);
    edges.duint.ensure_sizes(part_counts);
    edges.dB_on_rho.ensure_sizes(part_counts);
    edges.dpsi_on_ch.ensure_sizes(part_counts);
    edges.drho_dt.ensure_sizes(part_counts);
    edges.divB.ensure_sizes(part_counts);
    edges.curlB.ensure_sizes(part_counts);
    edges.dudtAV.ensure_sizes(part_counts);
    edges.dudtAR.ensure_sizes(part_counts);

    const Tscal pmass   = edges.gpart_mass.value;
    const Tscal alpha_u = edges.alpha_u.value;
    const Tscal beta_AV = edges.beta_AV.value;
    const Tscal sigma_mhd = edges.sigma_mhd.value;
    const Tscal mu_0 = edges.mu_0.value;

    using ComputeKernel = KernelUpdateDerivsVaryingAlphaAV_MHD<Tvec, SPHKernel>;

    // call the kernel for each patches with part_counts.get(id_patch) threads of patch id_patch
    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{
            edges.xyz.get_spans(),
            edges.hpart.get_spans(),
            edges.vxyz.get_spans(),
            edges.uint.get_spans(),
            edges.omega.get_spans(),
            edges.pressure.get_spans(),
            edges.cs.get_spans(),
            edges.alpha_AV.get_spans(),
            edges.B_on_rho.get_spans(),
            edges.psi_on_ch.get_spans(),
            edges.neigh_cache},
        sham::DDMultiRef{edges.axyz.get_spans(),
            edges.duint.get_spans(),
            edges.dB_on_rho.get_spans(),
            edges.dpsi_on_ch.get_spans(),
            edges.drho_dt.get_spans(),
            edges.divB.get_spans(),
            edges.curlB.get_spans(),
            edges.dudtAV.get_spans(),
            edges.dudtAR.get_spans()},
        part_counts,
        ComputeKernel{pmass, alpha_u, beta_AV, sigma_mhd, mu_0});
}

using namespace shammath;
template class shammodels::sph::modules::NodeUpdateDerivsVaryingAlphaAV_MHD<f64_3, M4>;
template class shammodels::sph::modules::NodeUpdateDerivsVaryingAlphaAV_MHD<f64_3, M6>;
template class shammodels::sph::modules::NodeUpdateDerivsVaryingAlphaAV_MHD<f64_3, M8>;

template class shammodels::sph::modules::NodeUpdateDerivsVaryingAlphaAV_MHD<f64_3, C2>;
template class shammodels::sph::modules::NodeUpdateDerivsVaryingAlphaAV_MHD<f64_3, C4>;
template class shammodels::sph::modules::NodeUpdateDerivsVaryingAlphaAV_MHD<f64_3, C6>;
