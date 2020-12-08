#include <iostream>
#include <random>
#include <cmath>
#include <bits/stdc++.h>

//#undef assert
//#define assert(...) (void(0))

using namespace std;

// Create the grid
using Vec = double;
constexpr int N = 10;
constexpr int NT = 50;
constexpr double J = 1;
bool grid[N][N][N];
constexpr int steps = 1e3;
constexpr double T0 = 2.5; // lowest temperature
constexpr double TRange = 3.5; // delta to highest temperature

//#define START_UP
#define DIMENSIONS 3

// Setup random number generator for float and integer distributions.
std::random_device rd;
//std::mt19937 e2(rd());
ranlux48_base e2(32);//rd());
std::uniform_real_distribution<> dist(0, 1);
std::uniform_int_distribution<> randint{0, N-1};
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

/// Set's a random value for the grid at x and y
inline void SetRandom(int x, int y, int z)
{
    grid[x][y][z] = randbool(e2);
}

/// Convenience function to get the spin at the given x and y
inline Vec Val(int x, int y, int z)
{
    return grid[x][y][z] ? 1 : -1;
}

/// Safer convenience function to get the spin at the given x and y, applying %N wrapping.
inline Vec SVal(int x, int y, int z)
{
    return grid[(x+N)%N][(y+N)%N][(z+N)%N] ? 1 : -1;
}

/// Flips the grid value at the given point.
inline void Flip(int x, int y, int z)
{
    grid[x][y][z] = !grid[x][y][z];
}

/// Returns the energy for the grid. Counts each neighbor pairing only once.
double Energy()
{
    double energy = 0;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
            {
                energy += -J*Val(x,y,z)*Val((x+1)%N,y,z);
                energy += -J*Val(x,y,z)*Val(x,(y+1)%N,z);
                energy += -J*Val(x,y,z)*Val(x,y,(z+1)%N);
            }
    return energy;
}

/// Returns the magnetization of the grid.
Vec Magnetization()
{
    Vec energy = 0;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                energy += Val(x,y,z);
    return energy;
}

/// Resets the grid. Get's starting magnetization and energy.
inline void ResetGrid(double& energy, Vec& magnetization)
{
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
#ifdef START_UP
                grid[x][y][z] = {1};
#else
                SetRandom(x,y,z);
#endif
    energy = Energy();
    magnetization = Magnetization();
}

/// Performs a step of the metropolis update algorithm.
void MetropolisUpdate(double beta, double& energy, Vec& magnetization)
{
    int x = RandInt();
    int y = RandInt();
    int z = RandInt();

    auto pt = Val(x,y,z);
    Vec neighbors[] = {SVal(x+1,y,z), SVal(x-1,y,z),
                       SVal(x,y+1,z), SVal(x,y-1,z),
                       SVal(x,y,z+1), SVal(x,y,z-1)};
    double dH = 0, dM = 0;
    // By doubling the values, we account for the energy counting each pair once, regardless of order,
    // with the energy +J to -J, for example, so dH would be -2J, not -J.
    for (auto n : neighbors)
        dH += 2*J*pt*n;

    // For decreasing energy (dH<0) we always accept as thresh > 1, so our random number will be less than it.
    // Otherwise we conditionally accept it based on the inverse temperature beta.
    double thresh = exp(-dH*beta);
    double p = Rand();
    if (p < thresh)
    {
        Flip(x,y,z);
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
    int x = RandInt();
    int y = RandInt();
    int z = RandInt();

    int frontierIdcs[N*N*N];
//    bool cluster[N][N][N] = {false,};
    bool cluster[N*N*N] = {false,};
    memset(cluster,false,sizeof cluster);

    // Initialize our frontier with the one point
    int frontierSize = 1;
    int frontierPos = 0;
    frontierIdcs[0] = x*N*N+y*N+z;

    struct xy
    {
        int x,y,z;
    };
    auto AddFrontier = [&](const xy& n)
    {
        if (frontierPos+frontierSize >= N*N*N)
        {
            memmove(frontierIdcs,frontierIdcs+frontierPos,frontierSize*sizeof(int));
            frontierPos=0;
            cout << "Reset frontier location \n";
        }
        frontierIdcs[frontierPos+frontierSize] = n.x*N*N+n.y*N+n.z;
        frontierSize += 1;
    };


    int iterations = 0;
    for (; frontierSize; ++frontierPos,--frontierSize)
    {
        // Get the cluster index, skip if it is already in the cluster.
        int i = frontierIdcs[frontierPos];
        int ix = i / N / N;
        int iy = (i/N) % N;
        int iz = i % N;
        if (cluster[i])
            continue;

        // Add this point to the cluster
        cluster[i] = true;

        // Compare to teh neighboring values
        auto v = Val(ix,iy,iz);
        for (xy n : {xy{1,0,0},xy{-1,0,0},xy{0,1,0},xy{0,-1,0},xy{0,0,1},xy{0,0,-1}})
        {
            n.x = (N+ix+n.x)%N;
            n.y = (N+iy+n.y)%N;
            n.z = (N+iz+n.z)%N;
            auto nv = Val(n.x,n.y,n.z);
            // If it is parallel
            if (v*nv > 0)
            {
                float p = (1 - exp(-2*J*beta));
                // With probability 1-exp(-2*J*beta) add it to the frontier
                if (Rand() <  p)
                    AddFrontier(n);
            }
        }
        // And now that we've finished with this point, go ahead and flip it.
        Flip(ix,iy,iz);

        iterations++;
    }

    // Recalculate energy and magnetization
    energy = Energy();
    magnetization = Magnetization();
}


double energies[NT][steps+1] = {0,};
Vec magnetizations[NT][steps] = {0,};



int main()
{
    cout << "Starting" << endl;
    for (int b = 0; b < NT; ++b)
    {
        double energy,magnetization;
        ResetGrid(energy,magnetization);
        double T = NT > 1 ? T0 + b*TRange/(NT-1) : T0;
        double beta = 1/T;
        energies[b][steps] = beta;
        for (int i = 0; i < steps; ++i)
        {
//            MetropolisUpdate(beta,energy,magnetization);
            WolffUpdate(beta,energy,magnetization);
            energies[b][i] = energy;
            magnetizations[b][i] = magnetization;
//            if (i % 1000 == 0 && abs(energy - Energy()) > 1e-5)
//            {
//                cout << "Failed with E = " << energy;
//                cout << " rather than E = " << Energy() << endl;
//                assert(false);
//            }
        }
    }
    cout << "Finished" << endl;

    // Write to file

    ofstream outE("energy.dat", ios::out | ios::binary);
    ofstream outM("magnetization.dat", ios::out | ios::binary);
    ofstream outP("params.txt", ios::out);
    if(!outE || !outM || !outP) {
      cout << "Cannot open file.";
      return 1;
     }

    outE.write((char *) &energies, sizeof energies);
    outM.write((char *) &magnetizations, sizeof magnetizations);

    outP << DIMENSIONS << " " << N << " " << NT << " " << steps << " " << T0 << " " << TRange;

    outE.close();
    outM.close();
    outP.close();

    system("python3 Plot.py");
    return 0;
}
