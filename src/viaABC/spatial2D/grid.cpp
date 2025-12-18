#include "grid.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

Grid::Grid(const std::vector<std::vector<int>>& initial,
           const Parameters& p)
    : rows(initial.size()),
      cols(initial.empty() ? 0 : initial[0].size()),
      data(rows * cols),
      next(rows * cols),
      params(p)
{
    if (rows == 0 || cols == 0)
        throw std::runtime_error("Grid must be non-empty");

    for (const auto& row : initial)
        if (row.size() != cols)
            throw std::runtime_error("Grid must be rectangular");

    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            data[idx(i, j)] = static_cast<uint8_t>(initial[i][j]);

    std::random_device rd;
    rng.seed(rd());
}

void Grid::simulate()
{
    if (params.dt <= 0.0)
        throw std::runtime_error("dt must be positive");

    const int steps =
        static_cast<int>((params.t_end - params.t0) / params.dt);
    if (steps <= 0) return;

    const double p_upreg   = 1.0 - std::exp(-params.alpha * params.dt);
    const double p_hotspot = 1.0 - std::exp(-params.beta  * params.dt);

    for (int t = 0; t < steps; ++t)
    {
        next = data;

        // STEP 1 — Y → G
        for (std::size_t k = 0; k < data.size(); ++k)
            if (data[k] == STATE_Y && dist(rng) < p_upreg)
                next[k] = STATE_G;

        // STEP 2a — G → H spontaneous
        for (std::size_t k = 0; k < data.size(); ++k)
            if (data[k] == STATE_G && dist(rng) < p_hotspot)
                next[k] = STATE_H;

        data.swap(next);

        // STEP 2b — neighbor-induced hotspot
        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                const std::size_t k = idx(i, j);
                if (data[k] != STATE_G) continue;

                bool has_hotspot = false;
                for (int di = -1; di <= 1 && !has_hotspot; ++di)
                for (int dj = -1; dj <= 1 && !has_hotspot; ++dj)
                {
                    if (di == 0 && dj == 0) continue;
                    const int ni = int(i) + di;
                    const int nj = int(j) + dj;
                    if (ni >= 0 && nj >= 0 &&
                        ni < int(rows) && nj < int(cols))
                    {
                        has_hotspot |=
                            (data[idx(ni, nj)] == STATE_H);
                    }
                }

                if (has_hotspot && dist(rng) < params.gamma)
                    next[k] = STATE_H;
            }
        }

        data.swap(next);
    }
}
