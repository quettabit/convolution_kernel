#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
using namespace std;
#define K 3

// declaration of constant memory where the fiter values are stored
__constant__ float cm[K*K];

__device__ void conv(const float* gm,
                        float* convolved,
                        int bh,
                        int bw,
                        int ih,
                        int iw,
                        int ch,
                        int cw,
                        int smH,
                        int smW,
                        int k,
                        float* sm,
                        int gID,
                        int tID,
                        int nT,
                        int rel_row,
                        int rel_col,
                        int nRows,
                        int stopPrefetchRowID,
                        int lastActiveThreadID) {

    for(int i=k; i<=nRows; i++)
    {
        /*
            ----prefetch a pixel value from GM and store it in register----

            all threads fetch the cell value immediately below to the current cell iteratively

            last thread in the block would fetch k cells immediately below the current cell

            boundary check would be needed for the blocks that act on the bottom most partition of the input image to prevent it from prefetching out of image values.
        */
        float reg;
        float *regArr = new float[k];
        if(i <= stopPrefetchRowID){
            reg = gm[i * iw + gID];
            if(tID == lastActiveThreadID){
                for(int j=1; j<=k-1; j++){
                    regArr[j] = gm[i * iw + gID + j];
                }
            }
        }
        // load k * k pixels above the current cell
        float *imgPixels = new float[k*k];
        for(int r=i-k; r<i; r++){
            for(int c=0; c<k; c++){
                /* translate the indices to [0,k] using r - (i-k) as imgPixels is of size k*k */
                imgPixels[(r-i+k)*k + c] = sm[r * smW + tID + c];
            }
        }
        /*multiply image pixel values with filter values (direct convolution) */
        float convolvedCell = 0.0;
        for(int c=0; c<k*k; c++){
            convolvedCell += cm[c]*imgPixels[c];
        }
        //place the convolvedCell value into convolvedMatrix
        int cID = ( ( (rel_row * bh) + (i-k) ) * cw )+( rel_col * nT )+tID;
        convolved[cID] = convolvedCell;
        __syncthreads();
        if(i <= stopPrefetchRowID){
            sm[i * smW + tID] = reg;
            if(tID == lastActiveThreadID){
                for(int j=1; j<=k-1; j++){
                    int sID = i *smW + tID + j;
                    sm[sID] = regArr[j];
                }
            }
        }
        __syncthreads();
    }


}

__global__ void conv_kernel(const float* gm,
                             float* convolved,
                             int bh,
                             int bw,
                             int ih,
                             int iw,
                             int ch,
                             int cw,
                             int smH,
                             int smW,
                             int k) {

    int tID = threadIdx.x;
    int bID = blockIdx.x;
    int nT = blockDim.x;
    int nB = gridDim.x;
    int nBx = iw / nT;
    //printf("num of blocks is %d\n", nB);
    //printf("nB in a row is %d\n", nBx);
    //check for right border or bottom border thread block
    bool isBottomBorder = false;
    bool isRightBorder = false;
    // bottom border thread block
    if(bID >= nB - nBx) {
        //printf("bID : %d is bottom border\n", bID);
        isBottomBorder = true;
    }
    // right border thread block
    if((bID+1) % nBx == 0){
        //printf("bID : %d is right border\n", bID);
        isRightBorder = true;
    }

    // ---------------- Load k rows from GM into SM ----------------------
    extern __shared__ float sm[];
    // rel_row and rel_col maps the Thread Block to appropriate position
    int rel_row = bID / nBx;
    int rel_col = bID % nBx;
    // (rel_row * bh * iw) covers all the cells before row_ids bh, 2bh, 3bh ..
    // gID finally maps threads to cells at rows 0, bh, 2bh, 3bh, ...
    int gID = (rel_row * bh * iw) + (rel_col * nT) + tID;

    for(int i=0; i<k; i++){

        int sID = i * smW + tID;
        sm[sID] = gm[i * iw + gID];
        /* if last thread in the block, it should fetch additional k-1 pixels
           in each row which are needed for computation of the convolution
        */
        if(!isRightBorder && tID == nT-1){
            for(int j=1; j<=k-1; j++){
                sID = i *smW + tID + j;
                sm[sID] = gm[i * iw + gID + j];
            }
        }

    }

    __syncthreads();

    if( !isBottomBorder && !isRightBorder ){
        int lastActiveThreadID = nT - 1;
        int nRows = bh + k - 1;
        int stopPrefetchRowID = nRows;
        conv( gm, convolved, bh, bw,
                ih, iw, ch, cw, smH, smW, k,
                sm, gID, tID, nT, rel_row, rel_col,
                nRows, stopPrefetchRowID, lastActiveThreadID );
    }
    else if( isBottomBorder && isRightBorder ){
        /* make the last k-1 threads in the block to be idle. as there is no convolution needed for them */
        if(tID < (nT - (k-1))){
            int nRows = bh;
            int stopPrefetchRowID = nRows - 1;
            int lastActiveThreadID = nT - k;
            conv( gm, convolved, bh, bw,
                    ih, iw, ch, cw, smH, smW, k,
                    sm, gID, tID, nT, rel_row, rel_col,
                    nRows, stopPrefetchRowID, lastActiveThreadID );
        }
    }
    else if( isBottomBorder ){
        int nRows = bh;
        int stopPrefetchRowID = nRows-1;
        int lastActiveThreadID = nT - 1;
        conv( gm, convolved, bh, bw,
                ih, iw, ch, cw, smH, smW, k,
                sm, gID, tID, nT, rel_row, rel_col,
                nRows, stopPrefetchRowID, lastActiveThreadID );


    }
    else if( isRightBorder ){
        /* make the last k-1 threads in the block to be idle. as there is no convolution needed for them */
        if(tID < (nT - (k-1))){
            int nRows = bh + k - 1;
            int stopPrefetchRowID = nRows;
            int lastActiveThreadID = nT - k;
            conv( gm, convolved, bh, bw,
                    ih, iw, ch, cw, smH, smW, k,
                    sm, gID, tID, nT, rel_row, rel_col,
                    nRows, stopPrefetchRowID, lastActiveThreadID );
        }

    }



}
int main(int argc, char **argv){
    /* set values for image dimensions, block dimensions, filter size, stride ..
       some of the constraints to keep in mind are
        1. the value of k(filter size) should be less than blcH and blcW
        2. stride value(s) should be 1
    */
    int imgH = 20;
    int imgW = 20;
    int blcH = 10;
    int blcW = 10;
    int k    = K;
    int s    = 1;
    int imgDims = imgH * imgW;
    int imgSize = imgDims * sizeof(float);
    // create host array that can hold pixel intensity values
    float *h_img = new float[imgDims];
    for(int i=0; i<imgDims; i++){
        h_img[i] = 1.0;
    }
    // create device array that can hold pixel intensity values in GPU GM
    float *d_img;
    cudaMalloc((void **) &d_img, imgSize );
    cudaMemcpy(d_img, h_img, imgSize, cudaMemcpyHostToDevice);
    // create filter and copy to constant memory
    int filterDims = k * k;
    int filterSize = filterDims * sizeof(float);
    float *filter = new float[filterDims];
    for(int i=0; i<filterDims; i++){
        filter[i] = 0.5;
    }
    cudaMemcpyToSymbol(cm, filter, filterSize);
    // create host and device array that holds the convoluted matrix
    int convH = ( (imgH - k) / s ) + 1;
    int convW = convH;
    int convDims = convH * convW;
    int convSize = convDims * sizeof(float);
    float *h_convolved = new float[convDims];
    for(int i=0; i<convDims; i++){
        h_convolved[i] = 0;
    }
    float *d_convolved;
    cudaMalloc((void **) &d_convolved, convSize);
    cudaMemcpy(d_convolved, h_convolved, convSize, cudaMemcpyHostToDevice);
    // calculate shared memory size
    int smH = blcH + k - 1;
    int smW = blcW + k - 1;
    int smSize = smH * smW * sizeof(float);
    // call the kernel
    conv_kernel<<<4, 10, smSize>>>(d_img, d_convolved,
                                    blcH, blcW,
                                    imgH, imgW,
                                    convH, convW,
                                    smH, smW,
                                    k);
    cudaMemcpy(h_convolved, d_convolved, convSize, cudaMemcpyDeviceToHost);
    for(int i=0; i<convH; i++){
        for(int j=0; j<convW; j++){
            cout<<h_convolved[i*convW +j]<<" ";
        }
        cout<<"\n";
    }
    cudaDeviceReset();
    delete h_img;
    delete h_convolved;
    return 0;
}
