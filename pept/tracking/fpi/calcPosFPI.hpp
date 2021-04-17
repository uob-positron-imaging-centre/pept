/**
 * Permission granted explicitly by Cody Wiggins in March 2021 to publish his code in the `pept`
 * library under the GNU v3.0 license. If you use the `pept.tracking.fpi` submodule, please cite
 * the following paper:
 *
 *      C. Wiggins et al. "A feature point identification method for positron emission particle
 *      tracking with multiple tracers," Nucl. Instr. Meth. Phys. Res. A, 843:22, 2017.
 *
 * The original author's copyright notice is included below. A sincere thank you for your work.
 */

//
//  calcPosFPI.h
//  MultiPEPT Code Snippets
//
//  Created by Cody Wiggins on 3/17/21.
//  Copyright © 2021 Cody Wiggins. All rights reserved.
//
//  Code adapted from MultiPEPT v.1.18 by Cody Wiggins
//
//  Function to read in voxel data and return locations of PEPT tracer particles, calculated by G-means method


#ifndef calcPosFPI_h
#define calcPosFPI_h

#include <cstdint>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cstdio>

#include "PeptStructures.hpp"

using namespace std;

//////////////////////////////////////
// Function Definitions:
//////////////////////////////////////

// Interface to calling `calcPosFPI` with simple C data structures
static inline double* calcPosFPIC(double *voxels, ssize_t length, ssize_t width, ssize_t depth, double w, double r,
                   double lldCounts, ssize_t *out_rows, ssize_t *out_cols);


// calculate particle positions with feature point tracking technique
static inline vector<point3> calcPosFPI(double***A,  ssize_t length, ssize_t width, ssize_t depth, ssize_t w, double r, double lldCounts2, vector<point3> &error);
// input: A: 3D array of LOR crossings (smoothed)
//        length: size of first dimension of A
//        width: size of 2nd dimension of A
//        depth: size of 3rd dimension of A
//        w: search range to be used in local maxima calculation
//        r: fraction of peak value used as threshold
//        lldCounts: A secondary lld to prevent assigning local maxima to voxels with very low values
//        error: blank vector to be used for error calculation
// output: Calculates positions of particles via local maxima calculation and centroid refinement in A.
//         Returns centroids of particles in vector<point3>
//         Returns uncertainties in "error" - Note: these are the standard deviation of fitted gaussians and have not divided by sqrt(N) for final uncertainty calc


// Supplemental Functions


// Determine if a given set of indices is in a list of possize_ts
static inline bool isIn(ssize_t i, ssize_t j, ssize_t k, vector<point3>list);
// input: i, j, k: Indices being searched for within the list
//        list: List of integer points in R3
// output: Returns true/false based on whether or not the the point (i,j,k) is in
//         the list.


// get a continuous 3-pt gaussian fit to data
static inline void contGaussFit(double guess[3], double A[3], double x[3]);
// input: guess: Intensity, meanX, sigX guesses (I, x0, sig)
//        A: intensities at positions i-1, i, i+1
//        x: positions of i-1, i, i+1
// output: manipulates guess into fittend, intensity, mean, and sigma


// get REF of a 3x4 matrix
static inline void REF34(double A[3][4]);
// input: A: 3x4 matrix
// output: Returns A in row-echelon form

    
    









//
/////////////////////////////
// Begin main function
/////////////////////////////
static inline double* calcPosFPIC(double *voxels, ssize_t length, ssize_t width, ssize_t depth,
                                  double w, double r, double lldCounts,
                                  ssize_t *out_rows, ssize_t *out_cols)
{
    if (length < 2 || width < 2 || depth < 2)
    {
        perror("[ERROR]: The input grid should be at least 2 voxels long, wide and deep, and there "
               "should be at least two particle positions.");
        return NULL;
    }

    // Pointer to pointer to pointer form of the input flattened `voxels`, for `calcPosFPI`
    double ***v3;
    double *out_points;
    vector<point3> points;
    vector<point3> error;

    ssize_t stride = width * depth;
    ssize_t i, j;

    // Create a 3-pointer equivalent form by storing pointers into the input flattened `voxels`
    v3 = (double***)malloc(sizeof(double**) * length);

    for (i = 0; i < length; ++i)
    {
        v3[i] = (double**)malloc(sizeof(double*) * length);

        for (j = 0; j < width; ++j)
            v3[i][j] = voxels + i * stride + j * depth;
    }

    // Run `calcPosFPI`
    points = calcPosFPI(v3, length, width, depth, w, r, lldCounts, error);

    // Copy `points` and `error` into `out_points` -> 6 columns
    out_points = (double*)malloc(sizeof(double) * points.size() * 6);
    *out_rows = points.size();
    *out_cols = 6;

    for (i = 0; i < (ssize_t)points.size(); ++i)
    {
        for (j = 0; j < 3; ++j)
        {
            out_points[i * 6 + j] = points[i].u[j];
            out_points[i * 6 + j + 3] = error[i].u[j];
        }
    }

    // Free allocated memory
    for (i = 0; i < length; ++i)
        free(v3[i]);

    free(v3);

    return out_points;
}





//
// calculate particle positions with feature point tracking technique
static inline vector<point3> calcPosFPI(double***A,  ssize_t length, ssize_t width, ssize_t depth, ssize_t w, double r, double lldCounts2, vector<point3> &error)
// input: A: 3D array of LOR crossings (smoothed)
//        length: size of first dimension of A
//        width: size of 2nd dimension of A
//        depth: size of 3rd dimension of A
//        w: search range to be used in local maxima calculation
//        r: fraction of peak value used as threshold
//        lldCounts: A secondary lld to prevent assigning local maxima to voxels with very low values
//        error: blank vector to be used for error calculation
// output: Calculates positions of particles via local maxima calculation and centroid refinement in A.
//         Returns centroids of particles in vector<point3>
//         Returns uncertainties in "error" - Note: these are the standard deviation of fitted gaussians and have not divided by sqrt(N) for final uncertainty calc
{
    
    // initialize answer
    vector<point3> means;
    
    //   cout<<"made calcPosFeature"<<endl;
    //cout<<w<<endl;
    
    // get maxVal of A
    double maxVal=-1;
    for (ssize_t i=w; i<length-w; i++){
        for (ssize_t j=w; j<width-w; j++){
            for (ssize_t k=w; k<depth-w; k++) {
                
                if (A[i][j][k]>maxVal){
                    maxVal=A[i][j][k];
                }
                
            }
        }
    }
    
    
    // now find all local maxima
    
    // initialize local maxima as "guesses"
    vector<ssize_t> guessX;
    vector<ssize_t> guessY;
    vector<ssize_t> guessZ;
    vector<double> guess;
    // keep track of ties
    vector<point3> tie;
    
    
    
    // neglect ends where smoothing (via convolution) aberrations may exist
    for (ssize_t i=w; i<length-w; i++){
        for (ssize_t j=w; j<width-w; j++){
            for (ssize_t k=w; k<depth-w; k++) {
                
                bool isNotMax=false;
                
                if (A[i][j][k]<(maxVal*r) || A[i][j][k]<=lldCounts2){continue;} // make sure it's above r-th percent
                // and above LLD
                
                
                if (isIn(i,j,k,tie)){continue;} // make sure its not a repeat of a tie
                
                vector<point3> potentialTies;
                
                // now search local 2w+1 width cube for greater intensity and ties
                for (ssize_t ii=i-w; ii<=i+w; ii++){
                    for (ssize_t jj=j-w; jj<=j+w; jj++){
                        for (ssize_t kk=k-w; kk<=k+w; kk++){
                            
                            if((ii==i)&&(jj==j)&&(kk==k)){continue;}
                            isNotMax = (A[ii][jj][kk]>A[i][j][k]);
                            
                            if (isNotMax){break;}
                            
                            if (A[ii][jj][kk]==A[i][j][k] && !isIn(ii, jj, kk, tie)) // see if there's a tie
                            {
                                point3 trashTie(ii,jj,kk);
                                potentialTies.push_back(trashTie);
                                //     cout<<"value: "<<A[ii][jj][kk]<<endl;
                                //     cout<<"indices: "<<ii<<"  "<<jj<<"  "<<kk<<endl;
                                //     cout<<"tie size: "<<tie.size()<<endl;
                            }
                        }
                        
                        if (isNotMax) {break;}
                    }
                    
                    if (isNotMax) {break;}
                }
                
                
                // identify and store local maxima
                if ((!isNotMax)&&(A[i][j][k]>=(maxVal*r))){
                    // note if it's a tie
                    for (ssize_t ii=0; ii<(ssize_t)potentialTies.size(); ii++)
                    {
                        tie.push_back(potentialTies[ii]);
                    }
                    guessX.push_back(i);
                    guessY.push_back(j);
                    guessZ.push_back(k);
                    guess.push_back(A[i][j][k]);
                }
                
                
            }
        }
    }

    // now we have initial guesses
    
    // take centroids
    
    for (ssize_t i=0; i<(ssize_t)guess.size(); i++){
        double meanX=0, meanY=0, meanZ=0;// diffX=0, diffY=0, diffZ=0;
        double sigX=0, sigY=0, sigZ=0; // standard deviations
        // double sumA=0;
        
        
        // do calc with gauss fit
        // fitting to 1-D gaussian of form A(x)=Ix*exp(-(x-meanX)^2/(2*sigX^2))
        ssize_t x1=guessX[i]-1, x2=guessX[i], x3=guessX[i]+1;
        ssize_t y1=guessY[i]-1, y2=guessY[i], y3=guessY[i]+1;
        ssize_t z1=guessZ[i]-1, z2=guessZ[i], z3=guessZ[i]+1;
        double IMeanSig[3]; // vector of guesses (and eventually answers) for intensity, mean, stdev
        double fitInt[3]; // will be intensities of fitting pixels
        double fitPos[3]; // will be positions of fitting pixels
        
        ////////////////////////
        // first X-direction
        ////////////////////////
        meanX=(x1*x1-x2*x2)*log(A[x2][y2][z2]/A[x3][y2][z2])-(x2*x2-x3*x3)*log(A[x1][y2][z2]/A[x2][y2][z2]);
        meanX/=(x1-x2)*log(A[x2][y2][z2]/A[x3][y2][z2])-(x2-x3)*log(A[x1][y2][z2]/A[x2][y2][z2]);
        meanX*=0.5;
        
        sigX=log(A[x3][y2][z2])-log(A[x2][y2][z2]);
        sigX/=(x2-meanX)*(x2-meanX)-(x3-meanX)*(x3-meanX);// right now this is actually sigX^2
        // use this value in intensity fit
        //double Ix=A[x2][y2][z2]*exp((x2-meanX)*(x2-meanX)/(2.0*sigX));
        // now get actual sigX
        sigX=sqrt(sigX);
        // use these as initial guesses for continuous gaussian fit
        //IMeanSig[0]=Ix; IMeanSig[1]=meanX; IMeanSig[2]=sigX;
        IMeanSig[0]=A[x2][y2][z2]; IMeanSig[1]=x2; IMeanSig[2]=1.0;
        fitInt[0]=A[x1][y2][z2]; fitInt[1]=A[x2][y2][z2]; fitInt[2]=A[x3][y2][z2];
        fitPos[0]=x1; fitPos[1]=x2; fitPos[2]=x3;
        
        
        contGaussFit(IMeanSig,fitInt,fitPos); // do fit
        
        //if(true)
        //if(false)
        if(abs(IMeanSig[1]-x2)<1.) // make sure it converged near the original maxima voxel
        {
            // cout<<meanX<<endl<<IMeanSig[1]<<endl;
            meanX=IMeanSig[1];
            sigX=IMeanSig[2];
            // cout<<"used new fit x\n";
        }
        //sigX/=sqrt(A[x1][y2][z2]+A[x2][y2][z2]+A[x3][y2][z2]);
        
        
        
        ////////////////////////
        // Then y-direction
        ////////////////////////
        meanY=(y1*y1-y2*y2)*log(A[x2][y2][z2]/A[x2][y3][z2])-(y2*y2-y3*y3)*log(A[x2][y1][z2]/A[x2][y2][z2]);
        meanY/=(y1-y2)*log(A[x2][y2][z2]/A[x2][y3][z2])-(y2-y3)*log(A[x2][y1][z2]/A[x2][y2][z2]);
        meanY*=0.5;
        
        sigY=log(A[x2][y3][z2])-log(A[x2][y2][z2]);
        sigY/=(y2-meanY)*(y2-meanY)-(y3-meanY)*(y3-meanY);// right now this is actually sigX^2
        // use this value in intensity fit
        //double Iy=A[x2][y2][z2]*exp((y2-meanY)*(y2-meanY)/(2.0*sigY));
        // now get actual sigX
        sigY=sqrt(sigY);
        // use these as initial guesses for continuous gaussian fit
        //IMeanSig[0]=Iy; IMeanSig[1]=meanY; IMeanSig[2]=sigY;
        IMeanSig[0]=A[x2][y2][z2]; IMeanSig[1]=y2; IMeanSig[2]=1.0;
        fitInt[0]=A[x2][y1][z2]; fitInt[1]=A[x2][y2][z2]; fitInt[2]=A[x2][y3][z2];
        fitPos[0]=y1; fitPos[1]=y2; fitPos[2]=y3;
        
        contGaussFit(IMeanSig,fitInt,fitPos); // do fit
        
        //if(true)
        //if(false)
        if(abs(IMeanSig[1]-y2)<1.) // make sure it converged near the original maxima voxel
        {
            meanY=IMeanSig[1];
            sigY=IMeanSig[2];
            //cout<<"used new fit y\n";
        }
        //sigY/=sqrt(A[x2][y1][z2]+A[x2][y2][z2]+A[x2][y3][z2]);
        
        
        ////////////////////////
        // and z-direction
        ////////////////////////
        meanZ=(z1*z1-z2*z2)*log(A[x2][y2][z2]/A[x2][y2][z3])-(z2*z2-z3*z3)*log(A[x2][y2][z1]/A[x2][y2][z2]);
        meanZ/=(z1-z2)*log(A[x2][y2][z2]/A[x2][y2][z3])-(z2-z3)*log(A[x2][y2][z1]/A[x2][y2][z2]);
        meanZ*=0.5;
        
        sigZ=log(A[x2][y2][z3])-log(A[x2][y2][z2]);
        sigZ/=(z2-meanZ)*(z2-meanZ)-(z3-meanZ)*(z3-meanZ);// right now this is actually sigX^2
        // use this value in intensity fit
        //double Iz=A[x2][y2][z2]*exp((z2-meanZ)*(z2-meanZ)/(2.0*sigZ));
        // now get actual sigX
        sigZ=sqrt(sigZ);
        // use these as initial guesses for continuous gaussian fit
        //IMeanSig[0]=Iz; IMeanSig[1]=meanZ; IMeanSig[2]=sigZ;
        IMeanSig[0]=A[x2][y2][z2]; IMeanSig[1]=z2; IMeanSig[2]=1.0;
        fitInt[0]=A[x2][y2][z1]; fitInt[1]=A[x2][y2][z2]; fitInt[2]=A[x2][y2][z3];
        fitPos[0]=z1; fitPos[1]=z2; fitPos[2]=z3;
        
        contGaussFit(IMeanSig,fitInt,fitPos); // do fit
        
        //if(true)
        //if(false)
        if(abs(IMeanSig[1]-z2)<1.) // make sure it converged near the original maxima voxel
        {
            meanZ=IMeanSig[1];
            sigZ=IMeanSig[2];
            // cout<<"used new fit z\n";
        }
        //sigZ/=sqrt(A[x2][y2][z1]+A[x2][y2][z2]+A[x2][y2][z3]);
        
        
        
        
        //assign this to means and then get error and put this in "error"
        point3 trashMean(meanX,meanY,meanZ);
        means.push_back(trashMean);
        
        // now get error;
        // will be stdev/sqrt(# of points)
        
        //reset sumA... just in case
        // sumA=0;
        
        point3 trashErr(sigX,sigY,sigZ);
        error.push_back(trashErr);
        
    }
    
    // tell us how many positions detected for
    // cout<<"number of particles: "<<means.size()<<endl;

    return means;
    
}
// end main function



//////////////////////////////////
//  Supplemental functions
//////////////////////////////////


// Determine if a given set of indices is in a list of points
static inline bool isIn(ssize_t i, ssize_t j, ssize_t k, vector<point3>list)
// input: i, j, k: Indices being searched for within the list
//        list: List of integer points in R3
// output: Returns true/false based on whether or not the the point (i,j,k) is in
//         the list.
{
    for (ssize_t ii=0; ii<(ssize_t)list.size(); ii++)
    {
        if(i==list[ii].u[0] && j==list[ii].u[1] && k==list[ii].u[2])
        {
            return true;
        }
    }
    
    return false;
}



// get a continuous 3-pt gaussian fit to data
static inline void contGaussFit(double guess[3], double A[3], double x[3])
// input: guess: Intensity, meanX, sigX guesses (I, x0, sig)
//        A: intensities at positions i-1, i, i+1
//        x: positions of i-1, i, i+1
// output: manipulates guess into fitted intensity, mean, and sigma
{
    double I=guess[0], x0=guess[1], sig=guess[2];
   
    
    ssize_t nTrials=100; // how many max steps before convergence... arbitrarily chosen
    double delxTol=0.0001; // tolerance of guess convergence... arbitrarily chosen
    double fTol=0.0001; // tolerance of function convergence... arbitrarily chosen
    
    double F[3] = {0., 0., 0.}; // function to be zeroed: F[i](I, x0, sig)
    double J[3][3]; // jacobian of F
    
    // now loop through until convergence
    for(ssize_t k=0; k<nTrials; k++)
    {
        // initialize delX
        double delx[3];
        for(ssize_t i=0; i<3; i++){delx[i]=0;}
        
        // first need to find F and J, evaluated at guess
        for(ssize_t i=0; i<3; i++)
        {
            //get F
            F[i]=erf((x[i]+0.5-x0)/(sqrt(2.0)*sig))-erf((x[i]-0.5-x0)/(sqrt(2.0)*sig));
            F[i]*=I*sig*sqrt(3.14159/2.0);
            F[i]-=A[i];
            
            // see if we've hit convergence of F
            if(abs(F[0])+abs(F[1])+abs(F[2])<fTol)
            {
                guess[0]=I; guess[1]=x0; guess[2]=sig;
                // cout<<"F converged, iteration: "<<k<<endl;
                // cout<<"absF="<<abs(F[0])+abs(F[1])+abs(F[2])<<endl;
                // cout<<guess[0]<<"\t"<<guess[1]<<"\t"<<guess[2]<<endl;
                return;
            }
            
            // now each component of J
            // I-component
            J[i][0]=erf((x[i]+0.5-x0)/(sqrt(2.0)*sig))-erf((x[i]-0.5-x0)/(sqrt(2.0)*sig));
            J[i][0]*=sig*sqrt(3.14159/2.0);
            
            // x0-component
            J[i][1]=(exp(-(x[i]-0.5-x0)*(x[i]-0.5-x0)/(2.0*sig*sig))-exp(-(x[i]+0.5-x0)*(x[i]+0.5-x0)/(2.0*sig*sig)))*I;
            
            // sig-component
            J[i][2]=(x[i]+0.5-x0)*exp(-(x[i]+0.5-x0)*(x[i]+0.5-x0)/(2.0*sig*sig));
            J[i][2]-=(x[i]-0.5-x0)*exp(-(x[i]-0.5-x0)*(x[i]-0.5-x0)/(2.0*sig*sig));
            J[i][2]*=-I/sig;
            J[i][2]+=J[i][0]*I/sig;
        }
        
        // now we want to solve an equation of the form J*delx=-F
        // use REF formulation
        // make augmented matrix
        double B[3][4];
        for(ssize_t i=0; i<3; i++){
            B[i][3]=-F[i];
            for (ssize_t j=0; j<3; j++){
                B[i][j]=J[i][j];
            }
        }
        
        // Get REF(B)
        REF34(B);
        
        
        // now solve
        // now solve for delx
        for (ssize_t i=2; i>=0; i--){
            delx[i]=B[i][3];
            for (ssize_t j=2; j>i; j--){
                delx[i]-=delx[j]*B[i][j];
            }
            delx[i]/=B[i][i];
            
        }
        
        // now make sure we're getting closer to a zero
        double G[3]; // value of function at new position
        double f=(F[0]*F[0]+F[1]*F[1]+F[2]*F[2])/2.0;
        for(ssize_t i=0; i<3; i++)
        {
            //get G
            G[i]=erf((x[i]+0.5-x0-delx[1])/(sqrt(2.0)*(sig+delx[2])))-erf((x[i]-0.5-x0-delx[1])/(sqrt(2.0)*(sig+delx[2])));
            G[i]*=(I+delx[0])*(sig+delx[2])*sqrt(3.14159/2.0);
            G[i]-=A[i];
        }
        double g=(G[0]*G[0]+G[1]*G[1]+G[2]*G[2])/2.0;
        
        while(g>f) //do this until new point is lower than previous point
        {
            //scale delx according to a polynomial fit to find the minimum
            for(ssize_t i=0; i<3; i++){
                delx[i]*=f/(f+g);
            }
            
            //re-evaluate g at this point
            for(ssize_t i=0; i<3; i++)
            {
                //get G
                G[i]=erf((x[i]+0.5-x0-delx[1])/(sqrt(2.0)*(sig+delx[2])))-erf((x[i]-0.5-x0-delx[1])/(sqrt(2.0)*(sig+delx[2])));
                G[i]*=(I+delx[0])*(sig+delx[2])*sqrt(3.14159/2.0);
                G[i]-=A[i];
            }
            g=(G[0]*G[0]+G[1]*G[1]+G[2]*G[2])/2.0;
        }
        
        
        
        // see if delx is converged
        if (abs(delx[0])+abs(delx[1])+abs(delx[2])<delxTol){
            guess[0]=I+delx[0]; guess[1]=x0+delx[1]; guess[2]=sig+delx[2];
            // cout<<"delx converged, iteration: "<<k<<endl;
            // cout<<"abs(delx)="<<abs(delx[0])+abs(delx[1])+abs(delx[2])<<endl;
            // cout<<guess[0]<<"\t"<<guess[1]<<"\t"<<guess[2]<<endl;
            return;
        }
        
        // update I, x0, sigx
        I+=delx[0]; x0+=delx[1]; sig+=delx[2];
        
        //  cout<<I<<"\t"<<x0<<"\t"<<sig<<endl;
        
        
    }
    
    // add final values to guess
    //  cout<<"no convergence!!!"<<endl;
    guess[0]=I; guess[1]=x0; guess[2]=sig;
    // cout<<guess[0]<<"\t"<<guess[1]<<"\t"<<guess[2]<<endl;
    
    return;
}


// get REF of a 3x4 matrix
static inline void REF34(double A[3][4])
// input: A: 3x4 matrix
// output: Returns A in row-echelon form
{
    // gaussian elimination algorithm
    double maxLead=0;
    ssize_t maxLeadPos=0;
    for (ssize_t i=0; i<3; i++)
    {
        // first get pivot
        maxLead=A[i][i];
        maxLeadPos=i;
        for (ssize_t j=i+1; j<3; j++)
        {
            if(A[j][i]>maxLead)
            {
                maxLead=A[j][i];
                maxLeadPos=j;
            }
            
        }
        if (maxLeadPos!=i)
        {
            for (ssize_t j=i; j<4; j++)
            {
                swap(A[i][j],A[maxLeadPos][j]);
            }
        }
        
        
        // divide by first element in row
        for (ssize_t j=3; j>=i; j--)
        {
            A[i][j]/=A[i][i];
        }
        
        
        // subtract ith equation from others
        for (ssize_t k=i+1; k<3; k++){
            for (ssize_t j=3; j>=i; j--)
            {
                A[k][j]-=A[i][j]*A[k][i];
            }
        }
        
    }
    
    return;
}




#endif /* calcPosFPI_h */
