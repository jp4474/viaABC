#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

/* ------------------ logsumexp ------------------ */
double logsumexp(const std::vector<double>& x) {
    double max_val = -std::numeric_limits<double>::infinity();
    for (double v : x)
        if (v > max_val) max_val = v;

    if (!std::isfinite(max_val))
        return max_val;

    double sum = 0.0;
    for (double v : x)
        sum += std::exp(v - max_val);

    return max_val + std::log(sum);
}

/* ------------------ main function ------------------ */
double calculate_particle_weight(
    py::array_t<double> theta,
    py::array_t<double> prev_particles,
    py::array_t<double> prev_weights,
    py::array_t<double> prev_cov,
    py::function prior_log_prob
) {
    /* ---- buffer access ---- */
    auto t = theta.unchecked<1>();
    auto particles = prev_particles.unchecked<2>();
    auto weights = prev_weights.unchecked<1>();
    auto cov = prev_cov.unchecked<2>();

    const int N = particles.shape(0);
    const int d = particles.shape(1);

    if (theta.shape(0) != d)
        throw std::runtime_error("theta dimension mismatch");

    /* ---- call prior ---- */
    double prior_lp = prior_log_prob(theta).cast<double>();
    if (!std::isfinite(prior_lp))
        return 0.0;

    /* ---- map covariance to Eigen ---- */
    Eigen::MatrixXd C(d, d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            C(i, j) = cov(i, j);

    /* ---- regularize covariance ---- */
    C += 1e-6 * Eigen::MatrixXd::Identity(d, d);

    /* ---- Cholesky ---- */
    Eigen::LLT<Eigen::MatrixXd> llt(C);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("Covariance matrix not SPD");

    /* ---- log(det(cov)) ---- */
    double log_det_cov = 0.0;
    const auto& L = llt.matrixL();
    for (int i = 0; i < d; ++i)
        log_det_cov += 2.0 * std::log(L(i, i));

    /* ---- constants ---- */
    const double log_norm_const =
        -0.5 * (d * std::log(2.0 * M_PI) + log_det_cov);

    /* ---- compute log weights ---- */
    std::vector<double> log_w(N);
    Eigen::VectorXd theta_vec(d);
    for (int i = 0; i < d; ++i)
        theta_vec(i) = t(i);

    for (int i = 0; i < N; ++i) {
        double w = weights(i);
        if (w <= 0.0)
            w = 1e-300;  // clamp

        Eigen::VectorXd mean(d);
        for (int j = 0; j < d; ++j)
            mean(j) = particles(i, j);

        Eigen::VectorXd diff = theta_vec - mean;
        double quad = diff.dot(llt.solve(diff));

        double logpdf = log_norm_const - 0.5 * quad;
        log_w[i] = std::log(w) + logpdf;

        if (!std::isfinite(log_w[i]))
            throw std::runtime_error("Non-finite log weight");
    }

    double lse = logsumexp(log_w);
    if (!std::isfinite(lse))
        return 0.0;

    double result = std::exp(prior_lp - lse);
    return std::isfinite(result) ? result : 0.0;
}

/* ------------------ pybind ------------------ */
PYBIND11_MODULE(particle_weight_cpp, m) {
    m.def(
        "calculate_particle_weight",
        &calculate_particle_weight,
        py::arg("theta"),
        py::arg("prev_particles"),
        py::arg("prev_weights"),
        py::arg("prev_cov"),
        py::arg("prior_log_prob"),
        "Stable ABC-SMC particle weight computation"
    );
}
