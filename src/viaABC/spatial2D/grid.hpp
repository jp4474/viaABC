#pragma once
#include <vector>
#include <random>
#include <string>

// ---- Cell State Constants ---- //
const int STATE_R = 0;
const int STATE_Y = 1;
const int STATE_B = 2;
const int STATE_X = 3;
const int STATE_G = 4;
const int STATE_H = 5;

// ---- Parameters for Simulation ---- //
struct Parameters {
    double alpha;
    double beta;
    double gamma;
    double dt;
    double t0;
    double t_end;
};

// ---- Grid Class ---- //
class Grid {
private:
    std::vector<std::vector<int>> data;
    int rows, cols;

    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;
    Parameters params;

public:
    // Load from text file
    Grid(const std::string &filename, const Parameters &params);

    // NEW: Construct from numpy (passed as vector<vector<int>>)
    Grid(const std::vector<std::vector<int>> &initial, const Parameters &params);

    void loadFromFile(const std::string &filename);
    void saveToFile(const std::string &filename);

    int countNeighbors(int i, int j, int state) const;

    void simulate();                   // one step

    // Python accessors
    std::vector<std::vector<int>> getGrid() const { return data; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    void updateGrid(const std::vector<std::vector<int>> &new_data) { data = new_data; }
};
