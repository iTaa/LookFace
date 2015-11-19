/*
 *	Copyright (c) 2013, Nenad Markus
 *	All rights reserved.
 *
 *	This is an implementation of the algorithm described in the following paper:
 *		N. Markus, M. Frljak, I. S. Pandzic, J. Ahlberg and R. Forchheimer,
 *		Object Detection with Pixel Intensity Comparisons Organized in Decision Trees,
 *		http://arxiv.org/abs/1305.4537
 *
 *	Redistribution and use of this program as source code or in binary form, with or without modifications, are permitted provided that the following conditions are met:
 *		1. Redistributions may not be sold, nor may they be used in a commercial product or activity without prior permission from the copyright holder (contact him at nenad.markus@fer.hr).
 *		2. Redistributions may not be used for military purposes.
 *		3. Any published work which utilizes this program shall include the reference to the paper available at http://arxiv.org/abs/1305.4537
 *		4. Redistributions must retain the above copyright notice and the reference to the algorithm on which the implementation is based on, this list of conditions and the following disclaimer.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *	IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// * `angle` is a number between 0 and 1 that determines the counterclockwise in-plane rotation of the cascade:
//		0.0f corresponds to 0 radians and 1.0f corresponds to 2*pi radians

#ifndef  __PCIO_H_
#define  __PCIO_H_

#include "../../../general_def.h"

#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))

#define SQR(x) ((x)*(x))

// hyperparameters
#define NRANDS 1024
#define MAX_N 2000000

namespace FacePico{
    class pico
    {
    private:
        uint64_t prngglobal;
        int N = 0;
        uint8_t* ppixels[MAX_N];
        int pdims[MAX_N][2]; // (nrows, ncols)

        int nbackground = 0;
        int background[MAX_N]; // i

        int nobjects = 0;
        int objects[MAX_N][4]; // (r, c, s, i)

        float tsr, tsc;
        int tdepth;
        int ntrees=0;

        int32_t tcodes[4096][1024];
        float luts[4096][1024];

        float thresholds[4096];

        int rs[2*MAX_N];
        int cs[2*MAX_N];
        int ss[2*MAX_N];
        int iinds[2*MAX_N];
        float tvals[2*MAX_N];
        float os[2*MAX_N];

    public:
        pico(){
            prngglobal = 0x12345678000fffffLL;
        };
        ~pico();
    public:
        /////////////////////////////////////////////////////////////////////////
        // Testing 

        int         find_objects( float rs[], float cs[], float ss[], float qs[], int maxndetections, void* cascade, float angle, 
                                    void* pixels, int nrows, int ncols, int ldim, float scalefactor, float stridefactor, float minsize, float maxsize );
        int         cluster_detections(float rs[], float cs[], float ss[], float qs[], int n);
        
        void        process_webcam_frames(std::string video_path, int & usepyr, void* cascade, float angle, int & minsize, int & maxsize, 
                                            float & scalefactor, float & stridefactor, float & qthreshold, int & noclustering, int & verbose);
        cv::Rect    process_image(cv::Mat frame, int draw, int & usepyr, void* cascade, float angle, int & minsize, int & maxsize, 
                                    float & scalefactor, float & stridefactor, float & qthreshold, int & noclustering, int & verbose);

        int         run_cascade(void* cascade, float* o, int r, int c, int s, void* vppixels, int nrows, int ncols, int ldim);
        int         run_rotated_cascade(void* cascade, float* o, int r, int c, int s, float a, void* vppixels, int nrows, int ncols, int ldim);

        float       get_overlap(float r1, float c1, float s1, float r2, float c2, float s2);
        void        ccdfs(int a[], int i, float rs[], float cs[], float ss[], int n);
        int         find_connected_components(int a[], float rs[], float cs[], float ss[], int n);
        
        /////////////////////////////////////////////////////////////////////////
        // Training 

        float       getticks();
        // multiply with carry PRNG
        uint32_t    mwcrand_r(uint64_t* state);    
        void        smwcrand(uint32_t seed);
        uint32_t    mwcrand();
        int         load_image(uint8_t* pixels[], int* nrows, int* ncols, FILE* file);
        int         load_training_data(char* path);
        // regression trees
        int         bintest(int32_t tcode, int r, int c, int sr, int sc, int iind);

        float       get_split_error(int32_t tcode, float tvals[], int rs[], int cs[], int srs[], int scs[], 
                                    int iinds[], double ws[], int inds[], int indsnum);
        int         split_training_data(int32_t tcode, float tvals[], int rs[], int cs[], int srs[], int scs[],
                                    int iinds[], double ws[], int inds[], int ninds);
        int         grow_subtree(int32_t tcodes[], float lut[], int nodeidx, int d, int maxd, float tvals[], 
                                    int rs[], int cs[], int srs[], int scs[], int iinds[], double ws[], int inds[], int ninds);
        int         grow_rtree(int32_t tcodes[], float lut[], int d, float tvals[], int rs[], int cs[], int srs[], 
                                    int scs[], int iinds[], double ws[], int n);

        int         load_cascade_from_file(const char* path);
        int         save_cascade_to_file(const char* path);
        float       get_tree_output(int i, int r, int c, int sr, int sc, int iind);
        int         classify_region(float* o, int r, int c, int s, int iind);
        int         learn_new_stage(float mintpr, float maxfpr, int maxntrees, float tvals[], int rs[], int cs[], 
                            int ss[], int iinds[], float os[], int np, int nn);
        float       sample_training_data(float tvals[], int rs[], int cs[], int ss[], 
                            int iinds[], float os[], int* np, int* nn);
        int         learn_with_default_parameters(char* trdata, char* dst);
        const char* howto();
        int         gen_test(int argc, char* argv[]);

    };
}


#endif