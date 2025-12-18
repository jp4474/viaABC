#pragma once

#include <vector>
#include <random>
#include <cstddef>

// ---- Cell States ---- //
enum CellState : uint8_t {
    STATE_R = 0,
    STATE_Y = 1,
    STATE_B = 2,
    STATE_X = 3,
    STATE_G = 4,
    STATE_H = 5
};

// ---- Parameters ---- //
struct Parameters {
    double alpha{0.0};
    double beta{0.0};
    double gamma{0.0};
    double dt{0.0};
    double t0{0.0};
    double t_end{0.0};
};

// ---- Grid ---- //
class Grid {
private:
    std::size_t rows{0}, cols{0};

    // row-major flattened storage
    std::vector<uint8_t> data;
    std::vector<uint8_t> next;

    std::mt19937 rng;
    std::uniform_real_distribution<double> dist{0.0, 1.0};

    Parameters params;

    inline std::size_t idx(std::size_t i, std::size_t j) const {
        return i * cols + j;
    }

public:
    Grid(const std::vector<std::vector<int>>& initial,
         const Parameters& params);

    void simulate();

    // ---- Accessors ---- //
    std::size_t getRows() const noexcept { return rows; }
    std::size_t getCols() const noexcept { return cols; }

    const std::vector<uint8_t>& raw() const noexcept { return data; }
};
