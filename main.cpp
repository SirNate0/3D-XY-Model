#include <iostream>
#include <random>
#define _USE_MATH_DEFINES
#include <cmath>
#include <bits/stdc++.h>

//#undef assert
//#define assert(...) (void(0))

using namespace std;

/// Minimal 2 component vector implementation for the XY spins
struct Vec2
{
    /// Components.
    double x,y;

    /// Dot product.
    double operator* (const Vec2& rhs) const
    {
        return x*rhs.x + y*rhs.y;
    }
    /// Uniformly scale.
    Vec2 operator* (double s) const
    {
        return {x*s,y*s};
    }
    /// Uniformly scale with other order of operands.
    friend Vec2 operator* (double s, const Vec2& b)
    {
        return {b.x*s,b.y*s};
    }
    /// Negate a vector.
    Vec2 operator-() const
    {
        return {-x,-y};
    }
    /// Add two vectors.
    Vec2 operator+ (const Vec2& rhs) const
    {
        return {x+rhs.x, y+rhs.y};
    }
    /// Add assign a vector.
    Vec2& operator+= (const Vec2& rhs)
    {
        x+= rhs.x;
        y+= rhs.y;
        return *this;
    }
    /// Reflect a vector along another vector (so over the plane perpendicular to that other vector).
    /// Assumes the other vector is a direction vector.
    Vec2 reflected(const Vec2& along) const
    {
        double dot = along * (*this);
        dot *= -2;
        auto delta = along*dot;
        auto out = (*this) + delta;
//        assert(abs(out.length()-1) < 1e-3); // Confirm that we have retained the unit length.
        return out;
    }
    /// Returns the length of the vector.
    double length() const
    {
        return sqrt(*this * *this);
    }

};

// ---------------
// Create the grid
// ---------------

// Typedef the vector so we could possibly switch to other dimensionality for our spin.
using Vec = Vec2;
// The energy scaling, the coefficient multiplied by the energy from two neighboring (unit length) spins.
constexpr double J = 1;
// The number of elements along one side of the grid.
constexpr int N = 30;
// The actual grid.
Vec2 grid[N][N][N];
// The number of temperatures to iterate over.
constexpr int NT = 50;
// How many steps to simulate.
constexpr int steps = 100e3;
// The lowest temperature to simulate.
constexpr double T0 = 0.5; // lowest temperature
// The range of temperatures to simulate (i.e. simulate steps in [T0,T0+TRange]
constexpr double TRange = 3.5; // delta to highest temperature

// Define this to have all spins start up, rather than in a random direction.
//#define START_UP
// The number of dimensions in the grid, used in writing the output parameters.
#define DIMENSIONS 3



// ------------------------------------------------------------------
// Setup random number generator for float and integer distributions.
// ------------------------------------------------------------------

// We use a fixed seed for repeatable results (for debugging), rather than seeding from the random device.
//std::random_device rd;
// Mersenne Twister is a better random number generator, but it's slower.
//std::mt19937 e2(rd());
// We want fast computations so we'll use a faster one that is moderately good.
ranlux48_base e2(32);//rd());
// Class to get a real (floating point) random number between [0,1).
std::uniform_real_distribution<> dist(0, 1);
// Class to get a real (integer) random number between [0,N).
std::uniform_int_distribution<> randint{0, N-1};
// Class to get a random boolean.
std::uniform_int_distribution<> randbool{0, 1};

/// Get random int [0,N).
inline int RandInt()
{
    return randint(e2);
}

/// Get random double [0,1).
inline double Rand()
{
    return dist(e2);
}

/// Get random vec2 direction.
inline Vec2 RandVec()
{
    double angle = Rand()*M_PI*2;
    return{cos(angle),sin(angle)};
}


// ---------------------------
// Grid interaction functions.
// ---------------------------

/// Set's a random value for the grid at x and y
inline void SetRandom(int x, int y, int z)
{
    grid[x][y][z] = RandVec();
}

/// Convenience function to get the spin at the given x and y
inline Vec Val(int x, int y, int z)
{
    return grid[x][y][z];
}

/// Safer convenience function to get the spin at the given x and y, applying %N wrapping.
inline Vec SVal(int x, int y, int z)
{
    return grid[(x+N)%N][(y+N)%N][(z+N)%N];
}

/// Flips the grid value at the given point.
inline void Flip(const Vec2& along, int x, int y, int z)
{
    grid[x][y][z] = grid[x][y][z].reflected(along);
}


// -----------------------------------
// Measurement functions for the grid.
// -----------------------------------

/// Returns the energy for the grid. Counts each neighbor pairing only once.
double Energy()
{
    double energy = 0;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
            {
                energy += -J*(Val(x,y,z)*Val((x+1)%N,y,z));
                energy += -J*(Val(x,y,z)*Val(x,(y+1)%N,z));
                energy += -J*(Val(x,y,z)*Val(x,y,(z+1)%N));
            }
    return energy;
}

/// Returns the magnetization of the grid.
Vec Magnetization()
{
    Vec mag{0,0};
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                mag += Val(x,y,z);
    return mag;
}


// -------------------------
// Initialization algorithm.
// -------------------------

/// Resets the grid. Get's starting magnetization and energy.
inline void ResetGrid(double& energy, Vec& magnetization)
{
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
            {
#ifdef START_UP
                grid[x][y][z] = {1,0};
#else
                SetRandom(x,y,z);
#endif
                assert(abs(grid[x][y][z].length()-1) < 1e-3);
            }

    energy = Energy();
    magnetization = Magnetization();
}

// ------------------
// Update algorithms.
// ------------------

/// Performs a step of the metropolis update algorithm.
void MetropolisUpdate(double beta, double& energy, Vec& magnetization)
{
    int x = RandInt();
    int y = RandInt();
    int z = RandInt();
    auto along = RandVec();

    auto pt = Val(x,y,z);
    Vec neighbors[] = {SVal(x+1,y,z), SVal(x-1,y,z),
                       SVal(x,y+1,z), SVal(x,y-1,z),
                       SVal(x,y,z+1), SVal(x,y,z-1)};
    double dH = 0;
    Vec2 dM{0,0};
    // By doubling the values, we account for the energy counting each pair once, regardless of order,
    // with the energy +J to -J, for example, so dH would be -2J, not -J.
    for (auto n : neighbors)
        dH += 2*J*(along*pt)*(along*n);

    // For decreasing energy (dH<0) we always accept as thresh > 1, so our random number will be less than it.
    // Otherwise we conditionally accept it based on the inverse temperature beta.
    double thresh = exp(-dH*beta);
    double p = Rand();
    if (p < thresh)
    {
        Flip(along,x,y,z);
        dM = -2*pt;
    }
    else
    {
        dH = 0;
    }

//    assert(abs(Energy()-energy - dH) < 1e-4);

    energy += dH;
    magnetization += dM;
}

/// Perform Wolff cluster update.
void WolffUpdate(double beta, double& energy, Vec& magnetization)
{
    // Get a random point and a random direction to flip along.
    int x = RandInt();
    int y = RandInt();
    int z = RandInt();
    Vec2 along = RandVec(); // Effectively, we use this vector to map the XY model to an Ising model with spins [-1,1] instead of just +/-1.

    /// Setup a frontier array of the next points that will be added to the grid. To avoid possibly
    /// costly memory moves, we use the grid size times a few and use it as a buffer that we only
    /// add elements further down the line on while we just move the starting position along the buffer.
    constexpr int MAX_FRONTIER = N*N*N*3;
    int frontierIdcs[MAX_FRONTIER];
    bool cluster[N*N*N] = {false,};
    memset(cluster,false,sizeof cluster);

    // Initialize our frontier with the randomly chosen point.
    int frontierSize = 1;
    int frontierPos = 0;
    frontierIdcs[0] = x*N*N+y*N+z;

    // Struct to hold xy indices for convenience.
    struct grid_indices
    {
        // The xyz indices.
        int x,y,z;
    };
    // Method to add a point to the frontier (the points that will be added to the cluster).
    auto AddFrontier = [&](const grid_indices& n)
    {
        if (frontierPos+frontierSize >= MAX_FRONTIER)
        {
            memmove(frontierIdcs,frontierIdcs+frontierPos,frontierSize*sizeof(int));
            frontierPos=0;
            cout << "Reset frontier location \n";
        }
        frontierIdcs[frontierPos+frontierSize] = n.x*N*N+n.y*N+n.z;
        frontierSize += 1;
    };

    // Counters for the number of iterations and rejected bonds, useful for debugging.
    int iterations = 0, rejections = 0;

    // As long as we have points in our frontier
    for (; frontierSize; ++frontierPos,--frontierSize)
    {
        // Get the cluster index, skip if it is already in the cluster, and convert back to the xyz indexing.
        int i = frontierIdcs[frontierPos];
        if (cluster[i])
            continue;
        int ix = i / N / N;
        int iy = (i/N) % N;
        int iz = i % N;

        // Add this point to the cluster
        cluster[i] = true;

        // Compare to the neighboring values
        auto v = Val(ix,iy,iz);
        for (grid_indices n : {grid_indices{1,0,0},grid_indices{-1,0,0},
                               grid_indices{0,1,0},grid_indices{0,-1,0},
                               grid_indices{0,0,1},grid_indices{0,0,-1}})
        {
            // Get the proper neighbor index from the relative indexing used above.
            n.x = (N+ix+n.x)%N;
            n.y = (N+iy+n.y)%N;
            n.z = (N+iz+n.z)%N;
            // Get the spin value at the point.
            auto nv = Val(n.x,n.y,n.z);
            // If it is parallel add it to the frontier with probability 1-exp(-2*J*beta*spin*spin_neighbor)
            // If it's not parallel, p < 0, so Rand() will never be less than it. Note that (along*v) is a dot product.
            float p = (1 - exp(-2*J*beta*(along*v)*(along*nv)));
            if (Rand() <  p)
                AddFrontier(n);
            else
                // Increment our rejectiosn count
                rejections++;
        }
        // And now that we've finished with this point, go ahead and flip it.
        Flip(along,ix,iy,iz);

        // Increment our iterations count.
        iterations++;
    }

    // Recalculate energy and magnetization.
    energy = Energy();
    magnetization = Magnetization();
}


// Initialize energy and magnetization vectors.
// Energy is padded by 1 to also output the inverse temperature as the final element.
double energies[NT][steps+1] = {0,};
Vec magnetizations[NT][steps] = {0,};


// The main loop.
int main()
{
    cout << "Starting" << endl;

    // Iterate over our temperatures.
    for (int b = 0; b < NT; ++b)
    {
        // Print as a status update.
        cout << b << " of " << NT << endl;

        // Initialize our grid for this temperature.
        double energy;
        Vec2 magnetization;
        ResetGrid(energy,magnetization);

        // Calculate the current temperature from [T0,T0+TRange].
        double T = NT > 1 ? T0 + b*TRange/(NT-1) : T0;
        // And beta from temperature.
        double beta = 1/T;
        // And store it in the energies vector (useful for debugging).
        energies[b][steps] = beta;

        // The update loop for this temperature.
        for (int i = 0; i < steps; ++i)
        {
            // The actual update
//            MetropolisUpdate(beta,energy,magnetization);
            WolffUpdate(beta,energy,magnetization);

            // Store the energy and magnetization in the arrays
            energies[b][i] = energy;
            magnetizations[b][i] = magnetization;

            // Debugging check.
//            if (i % 1000 == 0 && abs(energy - Energy()) > 1e-5)
//            {
//                cout << "Failed with E = " << energy;
//                cout << " rather than E = " << Energy() << endl;
//                assert(false);
//            }

            // 10% progress updates.
            if ((1+i)%(steps/10)==0)
                cout << "\t ITERATION " << i << endl;
        }
    }
    cout << "Finished" << endl;

    // Write to files. Binary for energy and magnetization, text for the parameters.
    ofstream outE("energy.dat", ios::out | ios::binary);
    ofstream outM("magnetization.dat", ios::out | ios::binary);
    ofstream outP("params.txt", ios::out);
    if(!outE || !outM || !outP) {
      cout << "Cannot open file.";
      return 1;
     }

    // Actually write the data.
    outE.write((char *) &energies, sizeof energies);
    outM.write((char *) &magnetizations, sizeof magnetizations);

    outP << DIMENSIONS << " " << N << " " << NT << " " << steps << " " << T0 << " " << TRange;

    // Close files (flushes the buffer, which must be done before calling the plot script).
    outE.close();
    outM.close();
    outP.close();

    // Attempt to call the python3 plot script to plot the calculated data.
    cout << "Output written. \nAttempting to call the python3 plot script." << endl;
    return system("python3 Plot.py");
}
