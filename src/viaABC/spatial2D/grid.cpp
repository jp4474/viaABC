#include "grid.hpp"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cmath>

// --------------------------------------------------
// Constructor: load from file
// --------------------------------------------------
Grid::Grid(const std::string &filename, const Parameters &p)
    : rows(0), cols(0), dist(0.0, 1.0), params(p)
{
    std::random_device rd;
    rng.seed(rd());
    loadFromFile(filename);
}

// --------------------------------------------------
// Constructor: initialize from vector<vector<int>>
// --------------------------------------------------
Grid::Grid(const std::vector<std::vector<int>> &initial, const Parameters &p)
    : rows(initial.size()),
      cols(initial.empty() ? 0 : initial[0].size()),
      data(initial),
      dist(0.0, 1.0),
      params(p)
{
    std::random_device rd;
    rng.seed(rd());
}

// --------------------------------------------------
// Load grid from text file
// --------------------------------------------------
void Grid::loadFromFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Failed to open grid file: " + filename);

    file >> rows >> cols;
    data.assign(rows, std::vector<int>(cols));

    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            file >> data[r][c];
}

// --------------------------------------------------
// Save grid to file
// --------------------------------------------------
void Grid::saveToFile(const std::string &filename)
{
    std::ofstream out(filename);
    out << rows << " " << cols << "\n";

    for (auto &row : data) {
        for (int v : row)
            out << v << " ";
        out << "\n";
    }
}

// --------------------------------------------------
// Neighbor count
// --------------------------------------------------
int Grid::countNeighbors(int i, int j, int state) const
{
    int count = 0;
    for (int di = -1; di <= 1; di++)
    {
        for (int dj = -1; dj <= 1; dj++)
        {
            if (di == 0 && dj == 0) continue; // skip self

            int ni = i + di;
            int nj = j + dj;

            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols)
            {
                if (data[ni][nj] == state)
                    count++;
            }
        }
    }
    return count;
}

// --------------------------------------------------
// Main simulation step
// --------------------------------------------------
void Grid::simulate()
{
    int steps = static_cast<int>((params.t_end - params.t0) / params.dt);

    for (int t = 0; t < steps; t++)
    {
        //----------------------------------------------------------------------
        // STEP 1 — Y → G upregulation
        //----------------------------------------------------------------------
        double p_upreg = 1.0 - std::exp(-params.alpha * params.dt);
        std::vector<std::vector<int>> next_grid = data;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (data[i][j] == STATE_Y)
                {
                    if (dist(rng) < p_upreg)
                        next_grid[i][j] = STATE_G;
                }
            }
        }
        
        data = next_grid;  // commit stage

        //----------------------------------------------------------------------
        // STEP 2 — G → H hotspot formation
        //----------------------------------------------------------------------
        double p_hotspot = 1.0 - std::exp(-params.beta * params.dt);
        next_grid = data;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (data[i][j] == STATE_G)
                {
                    if (dist(rng) < p_hotspot)
                        next_grid[i][j] = STATE_H;
                }
            }
        }

        data = next_grid;  // commit stage


        //----------------------------------------------------------------------
        // STEP 3 — hotspot dilation (G → H if any H in 8 neighbors)
        //----------------------------------------------------------------------
        next_grid = data;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (data[i][j] == STATE_G)
                {
                    int hotspot_neighbors = countNeighbors(i, j, STATE_H);

                    if (hotspot_neighbors > 0 && dist(rng) < params.gamma)
                    {
                        next_grid[i][j] = STATE_H;
                    }
                }
            }
        }

        data = next_grid;  // commit final stage
    }
}
