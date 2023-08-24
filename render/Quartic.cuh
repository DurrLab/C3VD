/*  Adapted from solve-quartic.cc in quartic repository.
    https://github.com/sidneycadot/quartic/
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>

/* Solves a * x^4 + b * x^3 + c * x^2 + d * x + e == 0. */
inline __device__
double solveQuartic(double a_, double b_, double c_, double d_, double e_)
{
    const thrust::complex<double> a = thrust::complex<double>(a_,0.0);
    const thrust::complex<double> b = thrust::complex<double>(b_,0.0) / a;
    const thrust::complex<double> c = thrust::complex<double>(c_,0.0) / a;
    const thrust::complex<double> d = thrust::complex<double>(d_,0.0) / a;
    const thrust::complex<double> e = thrust::complex<double>(e_,0.0) / a;

    const thrust::complex<double> Q1 = c * c - 3. * b * d + 12. * e;
    const thrust::complex<double> Q2 = 2. * c * c * c - 9. * b * c * d + 27. * d * d + 27. * b * b * e - 72. * c * e;
    const thrust::complex<double> Q3 = 8. * b * c - 16. * d - 2. * b * b * b;
    const thrust::complex<double> Q4 = 3. * b * b - 8. * c;

    const thrust::complex<double> Q5 = thrust::pow(Q2 / 2. + thrust::sqrt(Q2 * Q2 / 4. - Q1 * Q1 * Q1),1./3.);
    const thrust::complex<double> Q6 = (Q1 / Q5 + Q5) / 3.;
    const thrust::complex<double> Q7 = 2. * thrust::sqrt(Q4 / 12. + Q6);

    const thrust::complex<double> root0 = (-b - Q7 - thrust::sqrt(4. * Q4 / 6. - 4. * Q6 - Q3 / Q7)) / 4.;
    const thrust::complex<double> root1 = (-b - Q7 + thrust::sqrt(4. * Q4 / 6. - 4. * Q6 - Q3 / Q7)) / 4.;
    const thrust::complex<double> root2 = (-b + Q7 - thrust::sqrt(4. * Q4 / 6. - 4. * Q6 + Q3 / Q7)) / 4.;
    const thrust::complex<double> root3 = (-b + Q7 + thrust::sqrt(4. * Q4 / 6. - 4. * Q6 + Q3 / Q7)) / 4.;

    const double root0_r = root0.real();
    const double root1_r = root1.real();
    const double root2_r = root2.real();
    const double root3_r = root3.real();

    /* Find minimum real root greater than 0. */
    double solution = 1.0e50;
    if(root0.real() > 0.0 && (root0.imag() == 0.0))
        solution = min(solution,root0.real());
    if(root1.real() > 0.0 && (root1.imag() == 0.0))
        solution = min(solution,root1.real());
    if(root2.real() > 0.0 && (root2.imag() == 0.0))
        solution = min(solution,root2.real());
    if(root3.real() > 0.0 && (root3.imag() == 0.0))
        solution = min(solution,root3.real());
    if(solution == 1.0e50)
        solution = 0.0;

    return solution;
}