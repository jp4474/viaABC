#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <sstream>

// State constants
const int STATE_R = 0; // R cells (Red) - cells that can be activated
const int STATE_Y = 1; // Y cells (Yellow) - Activated with rate alpha
const int STATE_B = 2; // B cells (Blue) - these stay unchanged
const int STATE_X = 3; // X cells (no color) - these stay unchanged
const int STATE_G = 4; // G cells (Green) - converted from Y with upregulation rate beta
const int STATE_H = 5; // H cells (Hotspot) - formed from G cells with some probability

// Parameters structure
struct Parameters
{
    double alpha; // upgrade rate
    double beta;  // hotspot formation rate
    double gamma; // hotspot addition rate
    double dt;    // time step
    double t0;
    double t_end;
};

class Grid
{
private:
    std::vector<std::vector<int>> data;
    int rows, cols;
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;
    Parameters params;

public:
    Grid(const std::string &filename, const Parameters &p) : rows(0), cols(0), dist(0.0, 1.0), params(p)
    {
        std::random_device rd;
        rng.seed(rd());
        loadFromFile(filename);
    }

    void loadFromFile(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error opening file: " << filename << "\n";
            exit(1);
        }

        std::vector<std::vector<int>> temp_data;
        std::string line;

        while (std::getline(file, line))
        {
            std::vector<int> row;
            std::istringstream iss(line);
            int value;

            while (iss >> value)
            {
                row.push_back(value);
            }

            if (!row.empty())
            {
                temp_data.push_back(row);
            }
        }

        file.close();

        if (temp_data.empty())
        {
            std::cerr << "Error: empty grid file" << "\n";
            exit(1);
        }

        rows = temp_data.size();
        cols = temp_data[0].size();
        data = temp_data;

        std::cout << "Loaded grid: " << rows << " x " << cols << "\n";
    }

    void saveToFile(const std::string &filename)
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error opening file for writing: " << filename << "\n";
            return;
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                file << data[i][j];
                if (j < cols - 1)
                    file << " ";
            }
            file << "\n";
        }
        file.close();
        std::cout << "Saved final grid to: " << filename << "\n";
    }

    int countNeighbors(int row, int col, int state) const
    {
        int count = 0;
        for (int dr = -1; dr <= 1; dr++)
        {
            for (int dc = -1; dc <= 1; dc++)
            {
                if (dr == 0 && dc == 0)
                    continue;

                int nr = row + dr;
                int nc = col + dc;

                // Constant boundary condition (neighbors outside grid don't count)
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols)
                {
                    if (data[nr][nc] == state)
                        count++;
                }
            }
        }
        return count;
    }

    void simulate()
    {
        int steps = static_cast<int>((params.t_end - params.t0) / params.dt);
        std::cout << "Running simulation for " << steps << " steps..." << "\n";

        for (int t = 0; t < steps; t++)
        {
            // if ((t + 1) % 40 == 0)
            // {
            //     std::cout << "Step " << (t + 1) << "/" << steps << std::endl;
            // }

            std::vector<std::vector<int>> next_grid = data;

            // Step 1: Y -> G transitions (upregulation)
            double p_upreg = 1.0 - std::exp(-params.alpha * params.dt);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (data[i][j] == STATE_Y)
                    {
                        if (dist(rng) < p_upreg)
                        {
                            next_grid[i][j] = STATE_G;
                        }
                    }
                }
            }

            data = next_grid;

            // Step 2: G -> H transitions (hotspot formation)
            double p_hotspot = 1.0 - std::exp(-params.beta * params.dt);
            next_grid = data;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (data[i][j] == STATE_G)
                    {
                        if (dist(rng) < p_hotspot)
                        {
                            next_grid[i][j] = STATE_H;
                        }
                    }
                }
            }

            data = next_grid;

            // Step 3: Hotspot expansion (G cells neighboring H cells -> H)
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

            data = next_grid;
        }
    }

    int getRows() const { return rows; }
    int getCols() const { return cols; }
};

int main(int argc, char *argv[])
{
    std::cout << "=== Spatial Cellular Automaton Simulation ===" << "\n";

    Parameters params;
    std::string input_grid = "initial_grid_cpp.txt";
    std::string output_grid = "final_grid_cpp.txt";

    // Check if all the parameters are provided as command line arguments
    if (argc >= 8)
    {
        // Format: ./final_model alpha beta gamma dt t0 t_end input_grid [output_grid]
        params.alpha = std::stod(argv[1]);
        params.beta = std::stod(argv[2]);
        params.gamma = std::stod(argv[3]);
        params.dt = std::stod(argv[4]);
        params.t0 = std::stod(argv[5]);
        params.t_end = std::stod(argv[6]);
        input_grid = argv[7];
        if (argc >= 9)
            output_grid = argv[8];
    }
    else
    {
        // throw error if not enough arguments
        std::cerr << "Insufficient arguments provided. Usage:\n";
        std::cerr << "./final_model alpha beta gamma dt t0 t_end input_grid [output_grid]\n";
        std::cerr << "or\n";
        std::cerr << "./final_model [parameters.txt] [initial_grid.txt] [output_grid.txt]\n";
        return 1;
    }

    std::cout << "\nParameters:" << std::endl;
    std::cout << "  alpha (upgrade rate): " << params.alpha << std::endl;
    std::cout << "  beta (hotspot formation): " << params.beta << std::endl;
    std::cout << "  gamma (hotspot addition): " << params.gamma << std::endl;
    std::cout << "  dt (time step): " << params.dt << std::endl;
    std::cout << "  Time range: " << params.t0 << " to " << params.t_end << std::endl;
    std::cout << std::endl;

    Grid grid(input_grid, params);

    grid.simulate();

    grid.saveToFile(output_grid);

    std::cout << "\nSimulation complete!" << std::endl;
    // std::cout << "Usage: ./final_model alpha beta gamma dt t0 t_end input_grid [output_grid]" << std::endl;
    // std::cout << "   or: ./final_model [parameters.txt] [initial_grid.txt] [output_grid.txt]" << std::endl;

    return 0;
}
