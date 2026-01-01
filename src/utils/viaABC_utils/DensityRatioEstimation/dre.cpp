#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <cmath>
#include <numeric>
#include <vector>
#include <algorithm>
#include <iostream>

namespace py = pybind11;

// Helper to replicate np.array_split behavior for Cross Validation
std::vector<std::vector<int>> array_split(int length, int folds) {
    std::vector<std::vector<int>> result;
    int base = length / folds;
    int remainder = length % folds;
    int start = 0;
    for (int i = 0; i < folds; ++i) {
        int size = base + (i < remainder ? 1 : 0);
        std::vector<int> indices;
        for (int j = 0; j < size; ++j) {
            indices.push_back(start + j);
        }
        result.push_back(indices);
        start += size;
    }
    return result;
}

// Standalone Helper Functions
py::array_t<double> weighted_var(py::array_t<double> x_np, py::object weights_obj = py::none()) {
    auto x_buf = x_np.request();
    double* x_ptr = (double*)x_buf.ptr;
    int n = x_buf.shape[0];
    int d = (x_buf.ndim > 1) ? x_buf.shape[1] : 1; // Handling 1D or 2D

    Eigen::Map<Eigen::MatrixXd> x(x_ptr, d, n); // Eigen defaults to ColMajor, careful with shapes. 
    // Usually numpy is RowMajor. Let's map carefully. 
    // For simplicity in porting, let's treat input as standard RowMajor if 2D.
    // However, to ensure safety, we'll copy to an Eigen matrix handling RowMajor.
    Eigen::MatrixXd X = Eigen::MatrixXd::Map(x_ptr, n, d).transpose(); // (d, n)

    Eigen::VectorXd w(n);
    if (weights_obj.is_none()) {
        w.setOnes();
    } else {
        py::array_t<double> w_np = weights_obj.cast<py::array_t<double>>();
        auto w_buf = w_np.request();
        w = Eigen::Map<Eigen::VectorXd>((double*)w_buf.ptr, n);
    }

    double V1 = w.sum();
    double V2 = w.squaredNorm();

    // Weighted Mean
    Eigen::VectorXd xbar = (X * w) / V1;

    // Weighted Variance
    Eigen::VectorXd s2(d);
    for (int i = 0; i < d; ++i) {
        double num = 0.0;
        for (int j = 0; j < n; ++j) {
            num += w(j) * std::pow(X(i, j) - xbar(i), 2);
        }
        s2(i) = num / (V1 - (V2 / V1));
    }

    return py::array_t<double>(d, s2.data());
}

double calculate_densratio_basis_sigma(double sigma_1, double sigma_2) {
    return sigma_1 * sigma_2 / std::sqrt(std::abs(sigma_1 * sigma_1 - sigma_2 * sigma_2));
}

double weighted_sample_quantile(py::array_t<double> x_np, double alpha, py::object weights_obj = py::none()) {
    auto x_buf = x_np.request();
    int n = x_buf.shape[0];
    double* x_ptr = (double*)x_buf.ptr;
    std::vector<double> x(x_ptr, x_ptr + n);

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i1, int i2) { return x[i1] < x[i2]; });

    if (alpha == 0) return x[idx[0]];

    Eigen::VectorXd weights(n);
    if (weights_obj.is_none()) {
        weights.setOnes();
    } else {
        py::array_t<double> w_np = weights_obj.cast<py::array_t<double>>();
        weights = Eigen::Map<Eigen::VectorXd>((double*)w_np.request().ptr, n);
    }
    
    weights /= weights.sum();
    
    std::vector<double> cum_weights(n + 1, 0.0);
    for (int i = 0; i < n; ++i) {
        cum_weights[i+1] = cum_weights[i] + weights[idx[i]];
    }
    cum_weights.back() = 1.0; // Force exact 1.0 at end

    for (int i = 0; i < n; ++i) {
        if (cum_weights[i] < alpha && alpha <= cum_weights[i+1]) {
            return x[idx[i]];
        }
    }
    return x[idx[n-1]];
}

class DensityRatioEstimation {
public:
    int n_basis;
    double epsilon;
    int max_iter;
    double abs_tol;
    int conv_check_interval;
    int fold;
    bool optimize;
    double sigma = -1.0; // Negative indicates None

    Eigen::MatrixXd theta;
    Eigen::MatrixXd x_train;
    Eigen::VectorXd weights_x;
    Eigen::VectorXd weights_y;
    Eigen::VectorXd alpha; // The learned weights

    DensityRatioEstimation(int n=100, double epsilon=0.1, int max_iter=500, 
                           double abs_tol=0.01, int conv_check_interval=20, 
                           int fold=5, bool optimize=false)
        : n_basis(n), epsilon(epsilon), max_iter(max_iter), abs_tol(abs_tol),
          conv_check_interval(conv_check_interval), fold(fold), optimize(optimize) {}

    double gaussian_basis(const Eigen::VectorXd& x, const Eigen::VectorXd& center, double s) {
        return std::exp(-0.5 * (x - center).squaredNorm() / (s * s));
    }

    Eigen::MatrixXd compute_A(const Eigen::MatrixXd& x, double s) {
        int nx = x.rows();
        Eigen::MatrixXd A(nx, n_basis);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < n_basis; ++j) {
                A(i, j) = gaussian_basis(x.row(i), theta.row(j), s);
            }
        }
        return A;
    }

    std::pair<Eigen::VectorXd, Eigen::VectorXd> compute_b(const Eigen::MatrixXd& y, double s) {
        int ny = y.rows();
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n_basis);

        for (int i = 0; i < n_basis; ++i) {
            double sum_val = 0.0;
            for (int j = 0; j < ny; ++j) {
                sum_val += gaussian_basis(theta.row(i), y.row(j), s) * weights_y(j);
            }
            b(i) = sum_val;
        }
        
        double denom = b.squaredNorm();
        Eigen::VectorXd b_norm = b / denom;
        return {b, b_norm};
    }

    Eigen::VectorXd KLIEP(Eigen::MatrixXd& A, Eigen::VectorXd& b, Eigen::VectorXd& b_norm, Eigen::VectorXd& w_x, double s) {
        Eigen::VectorXd alp = Eigen::VectorXd::Constant(n_basis, 1.0 / n_basis);
        
        // Filter non-null A rows (equivalent to A > 1e-64)
        std::vector<int> valid_indices;
        for(int i=0; i<A.rows(); ++i) {
            if (A.row(i).maxCoeff() > 1e-64) valid_indices.push_back(i);
        }

        int n_valid = valid_indices.size();
        Eigen::MatrixXd A_full(n_valid, n_basis);
        Eigen::VectorXd w_x_full(n_valid);
        for(int i=0; i<n_valid; ++i) {
            A_full.row(i) = A.row(valid_indices[i]);
            w_x_full(i) = w_x(valid_indices[i]);
        }
        
        Eigen::VectorXd target_prev; 
        if (conv_check_interval > 0)
            target_prev = (A * alp); // simplified checking logic

        for (int iter = 0; iter < max_iter; ++iter) {
            Eigen::VectorXd Ax = A_full * alp;
            // Avoid division by zero
            Ax = Ax.cwiseMax(1e-12); 
            
            Eigen::VectorXd term = w_x_full.array() / Ax.array();
            Eigen::VectorXd dAdalpha = A_full.transpose() * term;
            
            alp += epsilon * dAdalpha;
            
            // Constraint projection
            double bTalpha = b.dot(alp);
            alp = (alp + (1.0 - bTalpha) * b_norm).cwiseMax(0.0);
            
            // Normalize
            bTalpha = b.dot(alp);
            if(bTalpha > 0) alp /= bTalpha;

            if (iter % conv_check_interval == 0) {
                // Compute full objective for convergence check on ALL x
                // Note: The original code re-calculates weighted_basis_sum on self.x
                Eigen::VectorXd target = (this->compute_A(this->x_train, s) * alp);
                if (iter > 0) {
                    if ((target - target_prev).norm() < abs_tol) break;
                }
                target_prev = target;
            }
        }
        return alp;
    }

    double KLIEP_lcv(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double s) {
        Eigen::MatrixXd A = compute_A(x, s);
        auto bs = compute_b(y, s);
        Eigen::VectorXd b = bs.first;
        Eigen::VectorXd b_norm = bs.second;

        std::vector<int> valid_idx;
        for(int i=0; i<A.rows(); ++i) {
            if (A.row(i).maxCoeff() > 1e-64) valid_idx.push_back(i);
        }

        if (valid_idx.empty()) return std::numeric_limits<double>::infinity();

        int n_valid = valid_idx.size();
        Eigen::MatrixXd A_full(n_valid, n_basis);
        Eigen::MatrixXd x_full(n_valid, x.cols());
        Eigen::VectorXd w_full(n_valid);
        
        for(int i=0; i<n_valid; ++i) {
            A_full.row(i) = A.row(valid_idx[i]);
            x_full.row(i) = x.row(valid_idx[i]);
            w_full(i) = weights_x(valid_idx[i]);
        }

        auto folds_idx = array_split(n_valid, fold);
        double total_score = 0.0;

        for (const auto& f_idx : folds_idx) {
            // Create training set (minus fold)
            std::vector<bool> is_in_fold(n_valid, false);
            for(int idx : f_idx) is_in_fold[idx] = true;

            int n_train = n_valid - f_idx.size();
            Eigen::MatrixXd A_train(n_train, n_basis);
            Eigen::VectorXd w_train(n_train);
            
            int train_counter = 0;
            for(int i=0; i<n_valid; ++i) {
                if(!is_in_fold[i]) {
                    A_train.row(train_counter) = A_full.row(i);
                    w_train(train_counter) = w_full(i);
                    train_counter++;
                }
            }

            Eigen::VectorXd alp = KLIEP(A_train, b, b_norm, w_train, s);

            // Calc score on fold
            double fold_score = 0.0;
            double weight_sum = 0.0;
            for(int idx : f_idx) {
                // prediction = A * alpha. We need A for this specific x
                // But we already have A_full computed.
                double pred = A_full.row(idx).dot(alp);
                if (pred <= 0) pred = 1e-12; // safety
                fold_score += std::log(pred) * w_full(idx);
                weight_sum += w_full(idx);
            }
            if (weight_sum > 0) total_score += (fold_score / weight_sum);
        }
        return total_score / fold;
    }

    void fit(py::array_t<double> x_np, py::array_t<double> y_np, 
             py::object w_x_obj = py::none(), py::object w_y_obj = py::none(), 
             py::object sigma_obj = py::none()) {
        
        auto x_buf = x_np.request();
        auto y_buf = y_np.request();
        int x_len = x_buf.shape[0];
        int y_len = y_buf.shape[0];
        int dim = x_buf.shape[1];

        // Copy to Eigen (RowMajor for compatibility with typical numpy 2d)
        x_train = Eigen::MatrixXd(x_len, dim);
        memcpy(x_train.data(), x_buf.ptr, sizeof(double) * x_len * dim);
        // Eigen defaults to ColMajor, so we map, copy, then transpose to get RowMajor storage logic in ColMajor struct
        // Actually simpler: Map as RowMajor, then assign to default MatrixXd
        using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        x_train = Eigen::Map<RowMatrixXd>((double*)x_buf.ptr, x_len, dim);
        Eigen::MatrixXd y_train = Eigen::Map<RowMatrixXd>((double*)y_buf.ptr, y_len, dim);

        if (x_len < n_basis) throw std::invalid_argument("n cannot be larger than number of samples.");

        theta = x_train.topRows(n_basis);

        if (w_x_obj.is_none()) weights_x = Eigen::VectorXd::Ones(x_len);
        else weights_x = Eigen::Map<Eigen::VectorXd>((double*)w_x_obj.cast<py::array_t<double>>().request().ptr, x_len);

        if (w_y_obj.is_none()) weights_y = Eigen::VectorXd::Ones(y_len);
        else weights_y = Eigen::Map<Eigen::VectorXd>((double*)w_y_obj.cast<py::array_t<double>>().request().ptr, y_len);

        weights_x /= weights_x.sum();
        weights_y /= weights_y.sum();

        // Sigma handling
        if (py::isinstance<py::float_>(sigma_obj)) {
            sigma = sigma_obj.cast<double>();
            optimize = false;
        } else if (py::isinstance<py::list>(sigma_obj) || py::isinstance<py::array>(sigma_obj)) {
            optimize = true; 
        }

        if (optimize) {
            if (!py::isinstance<py::list>(sigma_obj) && !py::isinstance<py::array>(sigma_obj))
                throw std::invalid_argument("To optimize, provide list of scales.");
            
            std::vector<double> candidates;
            if (py::isinstance<py::list>(sigma_obj)) {
                candidates = sigma_obj.cast<std::vector<double>>();
            } else {
                 auto s_buf = sigma_obj.cast<py::array_t<double>>().request();
                 candidates.assign((double*)s_buf.ptr, (double*)s_buf.ptr + s_buf.size);
            }
            
            double best_score = -std::numeric_limits<double>::infinity();
            double best_sigma = candidates[0];

            for (double s : candidates) {
                double score = KLIEP_lcv(x_train, y_train, s);
                if (score > best_score) {
                    best_score = score;
                    best_sigma = s;
                }
            }
            sigma = best_sigma;
        }

        if (sigma < 0) throw std::invalid_argument("Sigma must be provided.");

        Eigen::MatrixXd A = compute_A(x_train, sigma);
        auto bs = compute_b(y_train, sigma);
        alpha = KLIEP(A, bs.first, bs.second, weights_x, sigma);
    }

    py::array_t<double> predict(py::array_t<double> x_in) {
        auto buf = x_in.request();
        int n = buf.shape[0];
        int d = buf.shape[1];
        using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::MatrixXd X = Eigen::Map<RowMatrixXd>((double*)buf.ptr, n, d);
        
        Eigen::MatrixXd A = compute_A(X, sigma);
        Eigen::VectorXd res = A * alpha;
        return py::array_t<double>(n, res.data());
    }

    double max_ratio() {
        Eigen::MatrixXd A = compute_A(x_train, sigma);
        return (A * alpha).maxCoeff();
    }
};

PYBIND11_MODULE(dre_cpp, m) {
    m.doc() = "Density Ratio Estimation C++ backend";
    
    m.def("weighted_var", &weighted_var, py::arg("x"), py::arg("weights")=py::none());
    m.def("calculate_densratio_basis_sigma", &calculate_densratio_basis_sigma);
    m.def("weighted_sample_quantile", &weighted_sample_quantile, py::arg("x"), py::arg("alpha"), py::arg("weights")=py::none());

    py::class_<DensityRatioEstimation>(m, "DensityRatioEstimation")
        .def(py::init<int, double, int, double, int, int, bool>(),
             py::arg("n")=100, py::arg("epsilon")=0.1, py::arg("max_iter")=500,
             py::arg("abs_tol")=0.01, py::arg("conv_check_interval")=20,
             py::arg("fold")=5, py::arg("optimize")=false)
        .def("fit", &DensityRatioEstimation::fit,
             py::arg("x"), py::arg("y"), py::arg("weights_x")=py::none(), 
             py::arg("weights_y")=py::none(), py::arg("sigma")=py::none())
        .def("predict", &DensityRatioEstimation::predict)
        .def("max_ratio", &DensityRatioEstimation::max_ratio)
        .def_readwrite("sigma", &DensityRatioEstimation::sigma)
        .def_readwrite("optimize", &DensityRatioEstimation::optimize)
        .def_readonly("weights", &DensityRatioEstimation::alpha);
}
