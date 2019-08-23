/**
 * File              : birmingham_method_ext.c
 * License           : License: GNU v3.0
 * Author            : Sam Manger
 * Date              : 21.09.2019
 */

/*

 */

#include "birmingham_method_ext.h"

void birmingham_method_ext(const double *lines, double *location, double *used, unsigned int n, const double fopt)
{  

    /* Function receives a set of LORs from python formatted in:

        line[n] = [t, x1, y1, z1, x2, y2, z2]

        For the set of lines, the minimum distance point (MDP) is calculated.
        A number of lines that lie outside the standard deviation of the 
        MDP are then removed from the set, and the MDP is recalculated. This
        process is repeated until approximately a set fraction (fopt) of the
        original lines is left. The position is them returned.

        Requires:
            
            lines of response
            fopt

        Returns:

            tracked position
            used lines

    */

    double ttt[n], xx1[n], xx2[n], yy1[n], yy2[n], zz1[n], zz2[n];
    int use[n];
    double x12[n], y12[n], z12[n], r12[n], q12[n], p12[n], a12[n], b12[n], c12[n], d12[n], e12[n], f12[n],  r2[n], dev[n];
    // float tt,xx,yy,zz,er;

    double Con_factor = 150; // Convergence factor when removing LORs from set

    float x,y,z,error,avtime,avang,avpos,dx,dy,dz,dd,dismin,dismax;
    // double sumerr,sumx,sumy,sumz,sumx2,sumy2,sumz2;
    int imin, nused, i;
    // int ninc,nfail,Nmin,Ninitn,ipass,ivariable,ibinary,ifile=1;
    int ninit;

    long imax,nprev,nfin;
    // long ntries,nfirst,ltt,lxx,lyy,lzz,ler,ievent;

    double suma, sumb, sumc, sumd, sume, sumf, sump, sumq, sumr, ab, dq, dp, ar, ac, denom; // Initialise "sum" variables to be used for set of LORs
    
    void initarray() {

            // Initialise arrays for set of LORs, to be used in calculate

            // int i; // iterator

            int it;

            it = 0;

            for (it=0;it<ninit;it++)
            {

              // printf("%.2f\n", xx1[it] );
              // use[it]=1;
              x12[it]=xx1[it]-xx2[it];
              y12[it]=yy1[it]-yy2[it];
              z12[it]=zz1[it]-zz2[it];
              r2[it]=x12[it]*x12[it]+y12[it]*y12[it]+z12[it]*z12[it];

              // if(r2[it]==0)r2[it]=1e-6;

              r12[it]=y12[it]*z12[it]/r2[it];
              q12[it]=x12[it]*z12[it]/r2[it];
              p12[it]=x12[it]*y12[it]/r2[it];

              a12[it]=(y12[it]*y12[it]+z12[it]*z12[it])/r2[it];
              b12[it]=(x12[it]*x12[it]+z12[it]*z12[it])/r2[it];
              c12[it]=(y12[it]*y12[it]+x12[it]*x12[it])/r2[it];
              d12[it]=((yy2[it]*xx1[it]-yy1[it]*xx2[it])*y12[it]+(zz2[it]*xx1[it]-zz1[it]*xx2[it])*z12[it])/r2[it];
              e12[it]=((zz2[it]*yy1[it]-zz1[it]*yy2[it])*z12[it]+(xx2[it]*yy1[it]-xx1[it]*yy2[it])*x12[it])/r2[it];
              f12[it]=-((zz2[it]*yy1[it]-zz1[it]*yy2[it])*y12[it]+(zz2[it]*xx1[it]-zz1[it]*xx2[it])*x12[it])/r2[it];  
            }
    }

   

    // }

    void calculate(){
      
        suma=sumb=sumc=sumd=sume=sumf=sump=sumq=sumr=0;

        // printf("Initialised variables inside calculate()");
        
        for (i=0;i<ninit;i++)
        {
            // printf("(test)\n");
            if(use[i]==1){
                // Calculate "sum of" for lines in use
                suma=suma+a12[i];
                sumb=sumb+b12[i];
                sumc=sumc+c12[i];
                sumd=sumd+d12[i];
                sume=sume+e12[i];
                sumf=sumf+f12[i];
                sump=sump+p12[i];
                sumq=sumq+q12[i];
                sumr=sumr+r12[i];
             }
        }

        ab = suma*sumb-sump*sump;
        dq = sumd*sumq+suma*sumf;
        dp = sumd*sump+suma*sume;
        ar = suma*sumr+sumq*sump;
        ac = suma*sumc-sumq*sumq;
        denom =(ar*ar-ab*ac);

        if(denom == 0)
        {
            denom =1.0e-6;
        }
        if(ar==0)
        {
            ar=1.0e-6;
        }
        if(suma==0)
        {
            suma=1.0e-6;
        }

        z=(ab*dq+dp*ar)/denom;
        y=(z*ac+dq)/ar;
        x=(y*sump+z*sumq-sumd)/suma;

        error=avtime=avpos=0;

        //work out errors and time
        for (i=0;i<ninit;i++){
            dx=x-xx1[i];
            dy=y-yy1[i];
            dz=z-zz1[i];

            dd=(dx*z12[i]-dz*x12[i])*(dx*z12[i]-dz*x12[i]);
            dd=dd+(dy*x12[i]-dx*y12[i])*(dy*x12[i]-dx*y12[i]);
            dd=dd+(dz*y12[i]-dy*z12[i])*(dz*y12[i]-dy*z12[i]);
            dev[i]=dd/r2[i];
        //fprintf(output,"errors %d %d %f %f %f %f",i,use[i],dx,dy,dz,dev[i]);
            if(use[i]==1) {
                error+=dev[i];
                avtime+=ttt[i];
                }
            }
        error=sqrt(error/nused);
        avtime=avtime/nused;
        avang=avang/nused;
        if(avang>360)avang=avang-360;
        avpos=avpos/nused;
        //fprintf(output,"\n %f %f %f %f %f %f",x,y,z,error,suma,sumb);
    }


    void iterate(){
      while(ninit>0){

         calculate();
         // printf("done calculate: nfin: %d nused: %d \n",nfin,nused);
        
        if(nused==nfin){
            return;
        }

        nprev=nused;
        nused=0;

        for (i=0;i<ninit;i++){
            if(sqrt(dev[i])>(Con_factor*error/100))
              use[i]=0;
            else  {
              use[i]=1;
              nused=nused+1;
              }
        }

        //Have reduced to too few events, so restore closest unused events
         while(nused<nfin) {
            dismin=10000;
            for (i=0;i<ninit;i++)
              if(use[i]==0&&dev[i]<dismin){
                 imin=i;
                 dismin=dev[i];
                 }
            use[imin]=1;
            nused=nused+1;
        }

        //Haven't removed any, so remove furthest point
        while(nused>=nprev) {
            dismax=0;
            for (i=0;i<ninit;i++)
              if(use[i]==1&&dev[i]>dismax){
                 imax=i;
                 dismax=dev[i];
                 }
            use[imax]=0;
            nused=nused-1;
        }
       }
    }

    void result()
    {
        // printf("\n%10.1f\t%10.1f\t%10.1f\t%10.1f\t%10.1f\t%5d\n",avtime,x,y,z,error,nused);
        location[0] = avtime;
        location[1] = x;
        location[2] = y;
        location[3] = z;
        location[4] = error;
        location[5] = nused;

        for (i = 0; i < ninit; ++i)
        {
            used[i] = use[i];
        }
    }

    ninit = n;

    // printf("Righto...\n");

    // printf("ninit is: %i\n", ninit);


    i = 0;

    const double *line_i;

    for (i = 0; i < ninit; ++i)
    {
        // printf("About to do line number %i:\t", i);
        line_i = lines + i * 7;
        ttt[i] = line_i[0];
        xx1[i] = line_i[1];
        yy1[i] = line_i[2];
        zz1[i] = line_i[3];
        xx2[i] = line_i[4];
        yy2[i] = line_i[5];
        zz2[i] = line_i[6];
        use[i] = 1;
        // printf("%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n",ttt[i],xx1[i],yy1[i],zz1[i],xx2[i],yy2[i],zz2[i]);
        // printf("Getting x y and z i got to: %i\n", i);
    };

    //


    // printf("About to assign nused\n%i\n", it);
    
    nused = ninit;

    nfin = (n * fopt);//100;

    // printf("nused is %i", nused);

    //

    initarray();
    
    // printf("About to begin iterate");

    iterate();
    result();
    
}