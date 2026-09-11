# Streamfunction–vorticity cavity with single-site bosonic mean field

## Physical equations and sign convention

On the unit square, the lid at y=1 moves in positive x with U=1; ν=UL/Re=0.01. Define

$$u=\partial_y\psi,\qquad v=-\partial_x\psi,\qquad \omega=\partial_xv-\partial_yu=-\nabla^2\psi.$$

The steady equations are

$$R_\psi=\nabla^2\psi+\omega=0,$$

$$R_\omega=\nu\nabla^2\omega-\psi_y\omega_x+\psi_x\omega_y=0.$$

A right-moving lid produces negative streamfunction in the main clockwise vortex. No pressure variable is needed.

The grid has n=32 points in both directions **including walls**: x_i=i h, y_j=j h, h=1/(n−1). Arrays use `[j,i]=[y,x]`. The five-point Laplacian and centered first derivatives act at interior nodes. Advection is the centered advective form, with no upwinding, clipping, or added viscosity.

## Walls and the corner convention

Set ψ=0 at all walls, enforcing impermeability. With this constant wall value, the Thom wall-vorticity relations are

$$\omega_{0,i}=-2\psi_{1,i}/h^2,\qquad
\omega_{n-1,i}=-2\psi_{n-2,i}/h^2-2U/h,$$

$$\omega_{j,0}=-2\psi_{j,1}/h^2,\qquad
\omega_{j,n-1}=-2\psi_{j,n-2}/h^2.$$

For example, expand ψ one grid point below the top wall, using ψ_y=U and ψ=0 at the wall. The relation ψ(h below)≈−hU−h²ω_wall/2 gives the negative lid-forcing sign above. Thom's relation has local first-order wall-vorticity truncation error in general; the interior derivatives are second order. Spatial convergence is measured, not inferred solely from interior stencil order.

Wall ψ and ω are eliminated before constructing monomials. Thus only 2(n−2)² independent local kets exist. There is no post-step overwriting of an evolved field to repair boundary conditions. The prescribed velocities are used when reporting wall observables; interior velocities come from the measured streamfunction curl.

The moving lid and stationary side walls have incompatible tangential velocities at the two upper corner points. We assign zero reported velocity at the corners (stationary-wall convention), and average the two neighboring wall-vorticity values for plotting corner ω. **Corner ω never enters any interior stencil**, so that average cannot affect the solution. All comparisons use the same convention. Tiny secondary corner eddies are not expected to be resolved accurately on 32×32.

## Coupled artificial-time relaxation

To find the steady solution, use

$$\partial_\tau\psi=\kappa_\psi R_\psi,\qquad
\partial_\tau\omega=R_\omega,\qquad \kappa_\psi=0.1.$$

These two equations have exactly the desired steady roots for any positive κψ. They avoid an elliptic field solve: the streamfunction residual is reduced through local-state evolution alongside vorticity. They do **not** represent the physical startup evolution, because the Poisson constraint is not imposed exactly at each intermediate τ. In particular, τ=28 in the production run is a convergence coordinate, not a prediction of physical settling time. Changing κψ to 0.05 is included as a steady-root sensitivity check.

The iteration is not a proof of global convergence for arbitrary Reynolds number or grid. The defaults and acceptance checks are validated for the stated Re=100 case. The configurable maximum τ prevents an endless unconverged run.

## Bosonic fields and normally ordered generator

For each independent field/node r, store

$$|\chi_r\rangle=\sum_{m=0}^{N_b}c_{r,m}|m\rangle,\qquad
|\Psi\rangle=\bigotimes_r|\chi_r\rangle.$$

Local matrices are

$$a=\sum_{m=1}^{N_b}\sqrt{m}|m-1\rangle\langle m|,\qquad a^\dagger=a^T,\qquad I=I_{N_b+1}.$$

Measure α_r=⟨χ_r|a|χ_r⟩/⟨χ_r|χ_r⟩. The physical observables are ψ=sψ αψ and ω=sω αω, with sψ=100 and sω=10000. All production kets and coefficients are real. The implementation also supports complex operator algebra, and monitors imaginary expectation values.

Substitute these observables and the eliminated wall expressions into the finite-difference relaxation equations, and divide each equation by its target scale. This gives an affine-quadratic polynomial F(α):

$$F_{\psi,r}(\alpha)=\kappa_\psi\big[\Delta_h\alpha_{\psi,r}+(s_\omega/s_\psi)\alpha_{\omega,r}\big],$$

$$F_{\omega,r}(\alpha)=s_\omega^{-1}\big[\nu\Delta_h\omega-(D_y\psi)(D_x\omega)+(D_x\psi)(D_y\omega)\big].$$

Here all ψ and ω on the right are their affine expressions in α, including walls. In particular, the moving lid creates a constant term in Fω for the adjacent interior row. Assemble

$$\mathcal G=\sum_r a_r^\dagger F_r(a).$$

Every physical coefficient is divided by its target observable scale and multiplied by the scales of its source fields. Sparse linear arrays and quadratic index lists are coefficient storage for these normally ordered monomials. They are not a pressure matrix inverse or a field integrator.

## Full first-order mean-field factorization

Write every operator as its expectation plus a fluctuation, retaining only terms linear in fluctuations:

$$a_r^\dagger a_s\simeq\alpha_s a_r^\dagger+\alpha_r^*a_s-\alpha_r^*\alpha_s I,$$

$$a_r^\dagger a_sa_t\simeq\alpha_s\alpha_ta_r^\dagger+
\alpha_r^*\alpha_ta_s+\alpha_r^*\alpha_sa_t-2\alpha_r^*\alpha_s\alpha_tI.$$

The same rule applies to repeated-site factors; both annihilation contributions must be counted when s=t. Summing produces

$$\mathcal G_{MF}=\sum_j K_j,\qquad K_j=f_ja_j^\dagger+b_ja_j+c_jI,$$

$$f=F(\alpha),\qquad b_j=\sum_r\alpha_r^*\frac{\partial F_r}{\partial\alpha_j},\qquad c_j=-\alpha_jb_j.$$

Constant forcing contributes to f and not b. The full b terms are retained on both fields; the solver does not use a creation-only replacement. Choosing c_j as above distributes the total scalar factor among sites without changing the normalized evolution.

Evolve normalized local kets by

$$\frac{d|\chi_j\rangle}{d\tau}=(K_j-\operatorname{Re}\langle K_j\rangle I)|\chi_j\rangle.$$

The norm derivative is zero in exact arithmetic. RK4 advances the **Fock coefficients**, recomputing α, f, b, and c at every stage. Endpoint normalization controls integration roundoff; it is not a field correction. Vacuum kets encode the quiescent initial interior. No coherent-state reconstruction occurs after initialization.

For an ideal untruncated coherent ket, a|α⟩=α|α⟩ and the b term contributes only a scalar; differentiating the normalized annihilation expectation gives dα/dτ=f. Thus the coherent mean-field limit reproduces the stated nonlinear discrete equations. At finite cutoff or finite RK step this identity is approximate. That is why saved-state defects, cutoff/step/scale sensitivity, and independent physical residuals are essential checks.

The chosen observable scales keep occupations small and prevent non-Hermitian local evolution from amplifying integration contamination over the relaxation interval. They do not alter ν, Re, the lid speed, the mesh, or the physical steady root. The halved-scale run checks that numerical conditioning does not change the answer.

## What the verification establishes

1. Random-field tests compare the assembled polynomial against independent array stencils, including all four wall terms and lid forcing.
2. Jacobian tests check full first-order decoupling, including repeated-site monomials, and a coherent-tangent test checks the local state derivative.
3. Manufactured sine solutions check the reference Poisson inverse and its spatial refinement.
4. The saved-state verifier independently reconstructs ψ, ω, u, and v from local Fock vectors, verifies initialization and normalization, then evaluates both steady PDE residuals and discrete divergence.
5. Separate vacuum-start runs vary cutoff, RK step, observable scales, and κψ.
6. Independent DNS starts from zero interior vorticity, integrates physical time with SSPRK3, and uses a DST Dirichlet Poisson solve at every stage. Its module imports no production code and production imports no reference code.
7. Refined DNS and the published centerline data distinguish the bosonic/integration error from finite spatial resolution.

All residuals are maximum absolute values of the nondimensional equations. Velocity and interior-vorticity comparison norms exclude walls; no relative pointwise division is performed near zero crossings. Centerlines on the even 32×32 grid are linearly interpolated to x=0.5 or y=0.5.

Reference: U. Ghia, K. N. Ghia, and C. T. Shin, “High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method,” *Journal of Computational Physics* 48 (1982), 387–411, [doi:10.1016/0021-9991(82)90058-4](https://doi.org/10.1016/0021-9991(82)90058-4). Only the Re=100 numerical table entries needed for validation are included; the paper itself is not redistributed.
