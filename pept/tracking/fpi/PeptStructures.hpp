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
//  PeptStructures.h
//  MultiPEPT Code Snippets
//
//  Created by Cody Wiggins on 3/17/21.
//  Copyright © 2021 Cody Wiggins. All rights reserved.
//
//  Code adapted from MultiPEPT v.1.18 by Cody Wiggins
//
//  File for holding definition of structures used by MultiPEPT processing codes


#ifndef PeptStructures_h
#define PeptStructures_h

#include <vector>
#include <cstdlib>

using namespace std;

// structure for holding points in R3
struct point3{
    double u[3]; // contains x, y, z coordinates
    // using name "u" due to conventional vector notation
    point3(const point3& a) // copy constructor
    {
        u[0]=a.u[0];
        u[1]=a.u[1];
        u[2]=a.u[2];
    }
    point3(ssize_t x, ssize_t y, ssize_t z) // constructor with integers
    {
        u[0]=x; u[1]=y; u[2]=z;
    }
    point3(double x, double y, double z) // constructor with doubles
    {
        u[0]=x; u[1]=y; u[2]=z;
    }
    point3& operator=(point3& rhs) // copy assignment operator
    {
        if (this != &rhs)
        {
            u[0] = rhs.u[0];
            u[1] = rhs.u[1];
            u[2] = rhs.u[2];
        }

        return *this;
    }
};



// structure for cluster of 3d points
struct cluster{
    vector<point3> point; // contains a vector of point3's
    
    cluster(const cluster& a) // copy constructor
    {
        for (ssize_t i=0; i<(ssize_t)a.point.size(); i++){
            point.push_back(a.point[i]);
        }
    }
    cluster(point3 a) // constructor with single point3
    {
        point.push_back(a); // makes a single element vector
    }
    cluster(vector<point3> a) // constructor with vector of point3's
    {
        for (ssize_t i=0; i<(ssize_t)a.size(); i++)
        {
            point.push_back(a[i]);
        }
    }
    cluster& operator=(cluster& rhs) // copy assignment operator
    {
        if (this != &rhs)
            for (ssize_t i=0; i<(ssize_t)rhs.point.size(); i++)
                point.push_back(rhs.point[i]);

        return *this;
    }
};



// structure for holding points in R3 with time and error
// these are similar to point3's, but also contain time and uncertainty info
struct point3time{
    double u[3]; // x,y,z position
    double t;    // time
    double err[3]; // x,y,z uncertainty
    ssize_t nLOR;
    point3time(const point3time& a) // copy constructor
    {
        u[0]=a.u[0];
        u[1]=a.u[1];
        u[2]=a.u[2];
        t=a.t;
        err[0]=a.err[0];
        err[1]=a.err[1];
        err[2]=a.err[2];
        nLOR=a.nLOR;
    }
    point3time(ssize_t x, ssize_t y, ssize_t z, ssize_t time, ssize_t ex, ssize_t ey, ssize_t ez, ssize_t nLines) // integer constructor
    {
        u[0]=x; u[1]=y; u[2]=z; t=time; err[0]=ex; err[1]=ey; err[2]=ez; nLOR=nLines;
    }
    point3time(double x, double y, double z, double time, double ex, double ey, double ez, ssize_t nLines) // double constructor
    {
        u[0]=x; u[1]=y; u[2]=z; t=time; err[0]=ex; err[1]=ey; err[2]=ez; nLOR=nLines;
    }
    point3time(point3 x, point3 dx, int time, int nLines) // constructor with point3's and int time
    {
        u[0]=x.u[0]; err[0]=dx.u[0];
        u[1]=x.u[1]; err[1]=dx.u[1];
        u[2]=x.u[2]; err[2]=dx.u[2];
        t=time;
        nLOR=nLines;
    }
    point3time(point3 x, point3 dx, double time, ssize_t nLines) // constructor with point3's and double time
    {
        u[0]=x.u[0]; err[0]=dx.u[0];
        u[1]=x.u[1]; err[1]=dx.u[1];
        u[2]=x.u[2]; err[2]=dx.u[2];
        t=time;
        nLOR=nLines;
    }
    point3time& operator=(point3time& a) // copy assignment operator
    {
        if (this != &a)
        {
            u[0]=a.u[0];
            u[1]=a.u[1];
            u[2]=a.u[2];
            t=a.t;
            err[0]=a.err[0];
            err[1]=a.err[1];
            err[2]=a.err[2];
            nLOR=a.nLOR;
        }

        return *this;
    }
};



// structure for cluster of point3time's
// similar to "cluster", but using point3time's
struct clusterTime{
    vector<point3time> point; // contains vector of point3time
    ssize_t nLikely; // number to be used in post processing for filtering false positives
    
    clusterTime(const clusterTime& a) // copy constructor
    {
        for (ssize_t i=0; i<(ssize_t)a.point.size(); i++){
            point.push_back(a.point[i]);
        }
        nLikely=a.nLikely;
    }
    clusterTime(point3time a) // constructor with single point3time
    {
        point.push_back(a);
        nLikely=point.size();
    }
    clusterTime(vector<point3time> a) // constructor with vector of point3time
    {
        for (ssize_t i=0; i<(ssize_t)a.size(); i++)
        {
            point.push_back(a[i]);
        }
        nLikely=point.size();
    }
    clusterTime& operator=(clusterTime& a) // copy assignment operator
    {
        if (this != &a)
        {
            for (ssize_t i=0; i<(ssize_t)a.point.size(); i++){
                point.push_back(a.point[i]);
            }
            nLikely=a.nLikely;
        }

        return *this;
    }

};


#endif /* PeptStructures_h */
