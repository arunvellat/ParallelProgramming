// 
// Diffuse.cpp
//
// Monolythic C++ version of 1d diffusion with output.
//
// Compile with accompanying Makefile
//

#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <omp.h>

int main( int argc, char *argv[] ) 
{
  // Physical parameters 
  float x1 = -12;         // left most x value
  float x2 = 12;          // right most x value
  float D  = 1.0;         // diffusion constant

  // Simulation parameters 
  float runtime = 34.0;   // how long should the simulation try to compute?
  float outtime = 1.0;    // how often should the result be written to file?
  int numPoints = 256;    // how many points should there be on the x axis?

  // Parameters of the initial density 
  float a0 = 0.5 / M_PI;  // initial amplitude of the diffusive field
  float sigma0 = 1.0;     // initial spread of the diffusive field

  // Output files
  std::string filename = "data.dat";  // name of the file to which the results wil be written

   // Read the values from a file if specified on the command line
  if (argc > 1) {
    std::ifstream infile(argv[1]);
    infile >> x1;
    infile >> x2;
    infile >> D;
    infile >> numPoints;
    infile >> runtime;
    infile >> outtime;
    infile >> a0;
    infile >> sigma0;
    infile >> filename;
    infile.close();
  }

  // Compute derived parameters 
  float dx        = (x2 - x1) / (numPoints - 1);
  float dt        = dx * dx * D / 5;
  int   numSteps  = runtime/dt + 0.5;
  int   plotEvery = outtime/dt + 0.5;
  

  // Report all the values
  std::cout << "#x1        " << x1        << "\n";
  std::cout << "#x2        " << x2        << "\n";
  std::cout << "#D         " << D         << "\n";
  std::cout << "#numPoints " << numPoints << "\n";
  std::cout << "#runtime   " << runtime   << "\n";
  std::cout << "#outtime   " << outtime   << "\n";
  std::cout << "#a0        " << a0        << "\n";
  std::cout << "#sigma0    " << sigma0    << "\n";
  std::cout << "#filename  " << filename  << "\n";
   
  int rank, size, ierr, thread;
  int tagLeft =0, tagRight=1;
  float *xComplete, *rhoComplete; //for output
 
  ierr= MPI_Init(&argc, &argv);
   
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);

  //Find out the size of each cell
  int localSize;
  int leftover = numPoints%size;
  int cnts[size], offsets[size]; //to take care of MPI_Gatherv when numPoints is not divisible by size
 
  // compute local array size 
  for(int i=0; i < size; i++)
    if(i < leftover) cnts[i] = numPoints/size + 1;
    else cnts[i] = numPoints/size;

  localSize = cnts[rank];

  if(rank == 0) //to take care of output
    {
      xComplete = new float[numPoints];
      rhoComplete = new float[numPoints];
      offsets[0] = 0;
      for(int i=1; i < size; i++)
	offsets[i] = offsets[i-1]+cnts[i-1];
    }
  // Allocate the gridpoints
  float* x = new float[localSize + 2];
  
  // Allocate data, including ghost cells: before and active timestep.
  float* rho_0 = new float[localSize + 2];
  float* rho_1 = new float[localSize + 2];
  
  // Setup initial time
  float time = 0.0;
  
  // Setup grid
  // for(int i = 0; i < localSize + 2; i++) {
  // x[i] = x1 + i * dx;

  #pragma omp parallel for shared(x)
  for (int i = 0; i < localSize + 2; i++) {
      if(leftover==0)
        x[i] = x1 + (i)*dx  + rank*localSize*dx;
      else
	{
	  if(rank < leftover)
	    x[i] = x1 + (i)*dx + rank*localSize*dx;
	  else
	    x[i] = x1 + (i)*dx + rank*localSize*dx + leftover*dx;// + (rank-leftover)*localSize*dx;
	}
    }

  // Setup initial conditions for rho
#pragma omp parallel for shared(rho_1, x)
  for(int i = 0; i < localSize + 2; i++ ) 
    rho_1[i] = a0 * exp(-(pow(x[i],2)) / (2 * pow(sigma0,2)) );  
  
  std::ofstream dataFile(filename.c_str());
 
  MPI_Gatherv(x+1,cnts[rank], MPI_FLOAT, xComplete, cnts, offsets, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(rho_1+1,cnts[rank], MPI_FLOAT, rhoComplete, cnts, offsets, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
 //
 // Write out parameters and initial data
 //
 if(rank == 0) {
  
  dataFile << "#x1        " << x1        << "\n";
  dataFile << "#x2        " << x2        << "\n";
  dataFile << "#D         " << D         << "\n";
  dataFile << "#numPoints " << numPoints << "\n";
  dataFile << "#runtime   " << runtime   << "\n";
  dataFile << "#outtime   " << outtime   << "\n";
  dataFile << "#a0        " << a0        << "\n";
  dataFile << "#sigma0    " << sigma0    << "\n";
  dataFile << "#filename  " << filename  << "\n";  
  dataFile << "\n\n# t = " << time << "\n";

#pragma omp parallel for ordered schedule(dynamic)
  for(int i = 0; i < numPoints; i++ )
    #pragma omp ordered
    dataFile << xComplete[i] << " " << rhoComplete[i] << "\n";
  }
  // Time evolution
  for(int step = 1; step <= numSteps; step++ ) {
    
    float* temp  = rho_0;
    rho_0 = rho_1;
    rho_1 = temp;

    // Set up MPI communication status variables
    MPI_Request sendreq1, recvreq1, sendreq2, recvreq2;
    MPI_Status sendstat1, sendstat2, recvstat1, recvstat2;
	
    //Set up left and right neighbours
    int right = (rank==(size-1)?0:(rank+1));
    int left = (rank==0?(size-1):(rank-1));
         
    //Communicate to and from right
    ierr = MPI_Isend(&rho_0[1], 1, MPI_FLOAT, left, tagLeft, MPI_COMM_WORLD, &sendreq1);
    ierr = MPI_Irecv(&rho_0[0], 1, MPI_FLOAT, left, tagRight, MPI_COMM_WORLD,&recvreq1);
    ierr = MPI_Isend(&rho_0[localSize], 1, MPI_FLOAT, right, tagRight, MPI_COMM_WORLD,&sendreq2);
    ierr = MPI_Irecv(&rho_0[localSize+1], 1, MPI_FLOAT, right, tagLeft , MPI_COMM_WORLD,&recvreq2);
    
    //compute middle points
#pragma omp parallel for firstprivate(rho_0) shared(rho_1)
    for( int i = 2; i < localSize; i++ ) 
      rho_1[i] = rho_0[i] + dt*D/(dx*dx) * 
	(rho_0[i+1] + rho_0[i-1] - 2 * rho_0[i]);

    //Now wait for communication to finish
     
    ierr = MPI_Wait(&sendreq1, &sendstat1);
    ierr = MPI_Wait(&sendreq2, &sendstat2);
     
    ierr = MPI_Wait(&recvreq1, &recvstat1);
    ierr = MPI_Wait(&recvreq2, &recvstat2);
    
    //Compute the end points
    for (int i = 1; i < localSize+1; i+=localSize-1)
    {
      rho_1[i] = rho_0[i] + dt*D/(dx*dx) *
         (rho_0[i+1] + rho_0[i-1] - 2 * rho_0[i]);
    }

    time += dt;
    
     
     MPI_Gatherv(x+1,cnts[rank], MPI_FLOAT, xComplete, cnts, offsets, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Gatherv(rho_1+1,cnts[rank], MPI_FLOAT, rhoComplete, cnts, offsets, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // periodically add data to the file
     if(rank==0) {if( (step+1) % plotEvery == 0 ) { 
      std::cout << "Step = "  << step  << ", "
                << "Time = "  << time 
                << std::endl;
      dataFile << "\n\n# t = " << time << "\n";
     #pragma omp parallel for ordered schedule(dynamic)
      for(int i = 0; i < numPoints; i++ )
	#pragma omp ordered
	dataFile << xComplete[i] << " " << rhoComplete[i] << "\n";
      }}

  }
  
  // Close files
  dataFile.close();
  
  ierr = MPI_Finalize();

  // Free the memory used for the data
  if(rank == 0) { delete[] xComplete; delete[] rhoComplete; }
  delete[] rho_0;
  delete[] rho_1;
  delete[] x;
  
  return 0;
}
  
