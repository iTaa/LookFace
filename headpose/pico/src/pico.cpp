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



#include "pico.h"

using namespace FacePico;
using namespace cv;

int pico::run_cascade(void* cascade, float* o, int r, int c, int s, void* vppixels, int nrows, int ncols, int ldim)
{
	//
	int i, j, idx;

	uint8_t* pixels;

	float tsr, tsc;
	int tdepth, ntrees;

	int offset, sr, sc;

	int8_t* ptree;
	int8_t* tcodes;
	float* lut;
	float thr;

	//
	pixels = (uint8_t*)vppixels;

	//
	tsr = ((float*)cascade)[0];
	tsc = ((float*)cascade)[1];

	tdepth = ((int*)cascade)[2];
	ntrees = ((int*)cascade)[3];

	//
	sr = (int)(s*tsr);
	sc = (int)(s*tsc);

	r = r*256;
	c = c*256;

	if( (r+128*sr)/256>=nrows || (r-128*sr)/256<0 || (c+128*sc)/256>=ncols || (c-128*sc)/256<0 )
		return -1;

	//
	offset = ((1<<tdepth)-1)*sizeof(int32_t) + (1<<tdepth)*sizeof(float) + 1*sizeof(float);
	ptree = (int8_t*)cascade + 2*sizeof(float) + 2*sizeof(int);

	*o = 0.0f;

	for(i=0; i<ntrees; ++i)
	{
		//
		tcodes = ptree - 4;
		lut = (float*)(ptree + ((1<<tdepth)-1)*sizeof(int32_t));
		thr = *(float*)(ptree + ((1<<tdepth)-1)*sizeof(int32_t) + (1<<tdepth)*sizeof(float));

		//
		idx = 1;

		for(j=0; j<tdepth; ++j)
			idx = 2*idx + (pixels[(r+tcodes[4*idx+0]*sr)/256*ldim+(c+tcodes[4*idx+1]*sc)/256]<=pixels[(r+tcodes[4*idx+2]*sr)/256*ldim+(c+tcodes[4*idx+3]*sc)/256]);

		*o = *o + lut[idx-(1<<tdepth)];

		//
		if(*o<=thr)
			return -1;
		else
			ptree = ptree + offset;
	}

	//
	*o = *o - thr;

	return +1;
}

int pico::run_rotated_cascade(void* cascade, float* o, int r, int c, int s, float a, void* vppixels, int nrows, int ncols, int ldim)
{
	//
	int i, j, idx;

	uint8_t* pixels;

	float tsr, tsc;
	int tdepth, ntrees;

	int offset, sr, sc;

	int8_t* ptree;
	int8_t* tcodes;
	float* lut;
	float thr;

	static int qcostable[32+1] = {256, 251, 236, 212, 181, 142, 97, 49, 0, -49, -97, -142, -181, -212, -236, -251, -256, -251, -236, -212, -181, -142, -97, -49, 0, 49, 97, 142, 181, 212, 236, 251, 256};
	static int qsintable[32+1] = {0, 49, 97, 142, 181, 212, 236, 251, 256, 251, 236, 212, 181, 142, 97, 49, 0, -49, -97, -142, -181, -212, -236, -251, -256, -251, -236, -212, -181, -142, -97, -49, 0};

	//
	pixels = (uint8_t*)vppixels;

	//
	tsr = ((float*)cascade)[0];
	tsc = ((float*)cascade)[1];

	tdepth = ((int*)cascade)[2];
	ntrees = ((int*)cascade)[3];

	//
	r = r*65536;
	c = c*65536;

	if( (r+46341*s)/65536>=nrows || (r-46341*s)/65536<0 || (c+46341*s)/65536>=ncols || (c-46341*s)/65536<0 )
		return -1;

	//
	offset = ((1<<tdepth)-1)*sizeof(int32_t) + (1<<tdepth)*sizeof(float) + 1*sizeof(float);
	ptree = (int8_t*)cascade + 2*sizeof(float) + 2*sizeof(int);

	*o = 0.0f;

	int qsin = s*qsintable[(int)(32*a)]; //s*(int)(256.0f*sinf(2*M_PI*a));
	int qcos = s*qcostable[(int)(32*a)]; //s*(int)(256.0f*cosf(2*M_PI*a));

	for(i=0; i<ntrees; ++i)
	{
		//
		tcodes = ptree - 4;
		lut = (float*)(ptree + ((1<<tdepth)-1)*sizeof(int32_t));
		thr = *(float*)(ptree + ((1<<tdepth)-1)*sizeof(int32_t) + (1<<tdepth)*sizeof(float));

		//
		idx = 1;

		for(j=0; j<tdepth; ++j)
		{
			int r1, c1, r2, c2;

			//
			r1 = (r + qcos*tcodes[4*idx+0] - qsin*tcodes[4*idx+1])/65536;
			c1 = (c + qsin*tcodes[4*idx+0] + qcos*tcodes[4*idx+1])/65536;

			r2 = (r + qcos*tcodes[4*idx+2] - qsin*tcodes[4*idx+3])/65536;
			c2 = (c + qsin*tcodes[4*idx+2] + qcos*tcodes[4*idx+3])/65536;

			//
			idx = 2*idx + (pixels[r1*ldim+c1]<=pixels[r2*ldim+c2]);
		}

		*o = *o + lut[idx-(1<<tdepth)];

		//
		if(*o<=thr)
			return -1;
		else
			ptree = ptree + offset;
	}

	//
	*o = *o - thr;

	return +1;
}

int pico::find_objects	(
			float rs[], float cs[], float ss[], float qs[], int maxndetections,
			void* cascade, float angle, // * `angle` is a number between 0 and 1 that determines the counterclockwise in-plane rotation of the cascade: 0.0f corresponds to 0 radians and 1.0f corresponds to 2*pi radians
			void* pixels, int nrows, int ncols, int ldim,
			float scalefactor, float stridefactor, float minsize, float maxsize
		)
{
	float s;
	int ndetections;

	//
	ndetections = 0;
	s = minsize;

	while(s<=maxsize)
	{
		float r, c, dr, dc;

		//
		dr = dc = MAX(stridefactor*s, 1.0f);

		//
		for(r=s/2+1; r<=nrows-s/2-1; r+=dr)
			for(c=s/2+1; c<=ncols-s/2-1; c+=dc)
			{
				float q;
				int t;

				if(0.0f==angle)
					t = run_cascade(cascade, &q, r, c, s, pixels, nrows, ncols, ldim);
				else
					t = run_rotated_cascade(cascade, &q, r, c, s, angle, pixels, nrows, ncols, ldim);

				if(1==t)
				{
					if(ndetections < maxndetections)
					{
						qs[ndetections] = q;
						rs[ndetections] = r;
						cs[ndetections] = c;
						ss[ndetections] = s;

						//
						++ndetections;
					}
				}
			}

		//
		s = scalefactor*s;
	}

	//
	return ndetections;
}

/*
	
*/

float pico::get_overlap(float r1, float c1, float s1, float r2, float c2, float s2)
{
	float overr, overc;

	//
	overr = MAX(0, MIN(r1+s1/2, r2+s2/2) - MAX(r1-s1/2, r2-s2/2));
	overc = MAX(0, MIN(c1+s1/2, c2+s2/2) - MAX(c1-s1/2, c2-s2/2));

	//
	return overr*overc/(s1*s1+s2*s2-overr*overc);
}

void pico::ccdfs(int a[], int i, float rs[], float cs[], float ss[], int n)
{
	int j;

	//
	for(j=0; j<n; ++j)
		if(a[j]==0 && get_overlap(rs[i], cs[i], ss[i], rs[j], cs[j], ss[j])>0.3f)
		{
			//
			a[j] = a[i];

			//
			ccdfs(a, j, rs, cs, ss, n);
		}
}

int pico::find_connected_components(int a[], float rs[], float cs[], float ss[], int n)
{
	int i, ncc, cc;

	//
	if(!n)
		return 0;

	//
	for(i=0; i<n; ++i)
		a[i] = 0;

	//
	ncc = 0;
	cc = 1;

	for(i=0; i<n; ++i)
		if(a[i] == 0)
		{
			//
			a[i] = cc;

			//
			ccdfs(a, i, rs, cs, ss, n);

			//
			++ncc;
			++cc;
		}

	//
	return ncc;
}

int pico::cluster_detections(float rs[], float cs[], float ss[], float qs[], int n)
{
	int idx, ncc, cc;
	int a[4096];

	//
	ncc = find_connected_components(a, rs, cs, ss, n);

	if(!ncc)
		return 0;

	//
	idx = 0;

	for(cc=1; cc<=ncc; ++cc)
	{
		int i, k;

		float sumqs=0.0f, sumrs=0.0f, sumcs=0.0f, sumss=0.0f;

		//
		k = 0;

		for(i=0; i<n; ++i)
			if(a[i] == cc)
			{
				sumqs += qs[i];
				sumrs += rs[i];
				sumcs += cs[i];
				sumss += ss[i];

				++k;
			}

		//
		qs[idx] = sumqs; // accumulated confidence measure

		//
		rs[idx] = sumrs/k;
		cs[idx] = sumcs/k;
		ss[idx] = sumss/k;

		//
		++idx;
	}

	//
	return idx;
}


float pico::getticks()
{
	struct timespec ts;

	if(clock_gettime(CLOCK_MONOTONIC, &ts) < 0)
		return -1.0f;

	return ts.tv_sec + 1e-9f*ts.tv_nsec;
}

/*
	multiply with carry PRNG
*/

uint32_t pico::mwcrand_r(uint64_t* state)
{
	uint32_t* m;

	//
	m = (uint32_t*)state;

	// bad state?
	if(m[0] == 0)
		m[0] = 0xAAAA;

	if(m[1] == 0)
		m[1] = 0xBBBB;

	// mutate state
	m[0] = 36969 * (m[0] & 65535) + (m[0] >> 16);
	m[1] = 18000 * (m[1] & 65535) + (m[1] >> 16);

	// output
	return (m[0] << 16) + m[1];
}


void pico::smwcrand(uint32_t seed)
{
	prngglobal = 0x12345678000fffffLL*seed;
}

uint32_t pico::mwcrand()
{
	return mwcrand_r(&prngglobal);
}

int pico::load_image(uint8_t* pixels[], int* nrows, int* ncols, FILE* file)
{
	/*
	- loads an 8-bit grey image saved in the <RID> file format
	- <RID> file contents:
		- a 32-bit signed integer h (image height)
		- a 32-bit signed integer w (image width)
		- an array of w*h unsigned bytes representing pixel intensities
	*/

	//
	if(fread(nrows, sizeof(int), 1, file) != 1)
		return 0;

	if(fread(ncols, sizeof(int), 1, file) != 1)
		return 0;

	//
	*pixels = (uint8_t*)malloc(*nrows**ncols*sizeof(uint8_t));

	if(!*pixels)
		return 0;

	// read pixels
	if(fread(*pixels, sizeof(uint8_t), *nrows**ncols, file) != *nrows**ncols)
		return 0;

	// we're done
	return 1;
}

int pico::load_training_data(char* path)
{
	FILE* file;

	//
	file = fopen(path, "rb");

	if(!file)
		return 0;

	//
	N = 0;

	nbackground = 0;
	nobjects = 0;

	while( load_image(&ppixels[N], &pdims[N][0], &pdims[N][1], file) )
	{
		int i, n;

		//
		if(fread(&n, sizeof(int), 1, file) != 1)
			return 1;

		if(!n)
		{
			background[nbackground] = N;
			++nbackground;
		}
		else
		{
			for(i=0; i<n; ++i)
			{
				fread(&objects[nobjects][0], sizeof(int), 1, file); // r
				fread(&objects[nobjects][1], sizeof(int), 1, file); // c
				fread(&objects[nobjects][2], sizeof(int), 1, file); // s

				objects[nobjects][3] = N; // i

				//
				++nobjects;
			}
		}

		//
		++N;
	}

	//
	return 1;
}

/*
	regression trees
*/

int pico::bintest(int32_t tcode, int r, int c, int sr, int sc, int iind)
{
	//
	int r1, c1, r2, c2;
	int8_t* p = (int8_t*)&tcode;

	//
	r1 = (256*r + p[0]*sr)/256;
	c1 = (256*c + p[1]*sc)/256;

	r2 = (256*r + p[2]*sr)/256;
	c2 = (256*c + p[3]*sc)/256;

	//
	r1 = MIN(MAX(0, r1), pdims[iind][0]-1);
	c1 = MIN(MAX(0, c1), pdims[iind][1]-1);

	r2 = MIN(MAX(0, r2), pdims[iind][0]-1);
	c2 = MIN(MAX(0, c2), pdims[iind][1]-1);

	//
	return ppixels[iind][r1*pdims[iind][1]+c1]<=ppixels[iind][r2*pdims[iind][1]+c2];
}

float pico::get_split_error(int32_t tcode, float tvals[], int rs[], int cs[], int srs[], int scs[], int iinds[], double ws[], int inds[], int indsnum)
{
	int i, j;

	double wsum, wsum0, wsum1;
	double wtvalsum0, wtvalsumsqr0, wtvalsum1, wtvalsumsqr1;

	double wmse0, wmse1;

	//
	wsum = wsum0 = wsum1 = wtvalsum0 = wtvalsum1 = wtvalsumsqr0 = wtvalsumsqr1 = 0.0;

	for(i=0; i<indsnum; ++i)
	{
		if( bintest(tcode, rs[inds[i]], cs[inds[i]], srs[inds[i]], scs[inds[i]], iinds[inds[i]]) )
		{
			wsum1 += ws[inds[i]];
			wtvalsum1 += ws[inds[i]]*tvals[inds[i]];
			wtvalsumsqr1 += ws[inds[i]]*SQR(tvals[inds[i]]);
		}
		else
		{
			wsum0 += ws[inds[i]];
			wtvalsum0 += ws[inds[i]]*tvals[inds[i]];
			wtvalsumsqr0 += ws[inds[i]]*SQR(tvals[inds[i]]);
		}

		wsum += ws[inds[i]];
	}

	//
	wmse0 = wtvalsumsqr0 - SQR(wtvalsum0)/wsum0;
	wmse1 = wtvalsumsqr1 - SQR(wtvalsum1)/wsum1;

	//
	return (float)( (wmse0 + wmse1)/wsum );
}

int pico::split_training_data(int32_t tcode, float tvals[], int rs[], int cs[], int srs[], int scs[], int iinds[], double ws[], int inds[], int ninds)
{
	int stop;
	int i, j;

	int n0;

	//
	stop = 0;

	i = 0;
	j = ninds - 1;

	while(!stop)
	{
		//
		while( !bintest(tcode, rs[inds[i]], cs[inds[i]], srs[inds[i]], scs[inds[i]], iinds[inds[i]]) )
		{
			if( i==j )
				break;
			else
				++i;
		}

		while( bintest(tcode, rs[inds[j]], cs[inds[j]], srs[inds[j]], scs[inds[j]], iinds[inds[j]]) )
		{
			if( i==j )
				break;
			else
				--j;
		}

		//
		if( i==j )
			stop = 1;
		else
		{
			// swap
			inds[i] = inds[i] ^ inds[j];
			inds[j] = inds[i] ^ inds[j];
			inds[i] = inds[i] ^ inds[j];
		}
	}

	//
	n0 = 0;

	for(i=0; i<ninds; ++i)
		if( !bintest(tcode, rs[inds[i]], cs[inds[i]], srs[inds[i]], scs[inds[i]], iinds[inds[i]]) )
			++n0;

	//
	return n0;
}

int pico::grow_subtree(int32_t tcodes[], float lut[], int nodeidx, int d, int maxd, float tvals[], int rs[], int cs[], int srs[], int scs[], int iinds[], double ws[], int inds[], int ninds)
{
	int i, nrands;

	int32_t tmptcodes[2048];
	float es[2048], e;

	int n0;

	//
	if(d == maxd)
	{
		int lutidx;
		double tvalaccum, wsum;

		//
		lutidx = nodeidx - ((1<<maxd)-1);

		// compute output: a simple average
		tvalaccum = 0.0;
		wsum = 0.0;

		for(i=0; i<ninds; ++i)
		{
			tvalaccum += ws[inds[i]]*tvals[inds[i]];
			wsum += ws[inds[i]];
		}

		if(wsum == 0.0)
			lut[lutidx] = 0.0f;
		else
			lut[lutidx] = (float)( tvalaccum/wsum );

		//
		return 1;
	}
	else if(ninds <= 1)
	{
		//
		tcodes[nodeidx] = 0;

		//
		grow_subtree(tcodes, lut, 2*nodeidx+1, d+1, maxd, tvals, rs, cs, srs, scs, iinds, ws, inds, ninds);
		grow_subtree(tcodes, lut, 2*nodeidx+2, d+1, maxd, tvals, rs, cs, srs, scs, iinds, ws, inds, ninds);

		return 1;
	}

	// generate binary test codes
	nrands = NRANDS;

	for(i=0; i<nrands; ++i)
		tmptcodes[i] = mwcrand();

	//
	#pragma omp parallel for
	for(i=0; i<nrands; ++i)
		es[i] = get_split_error(tmptcodes[i], tvals, rs, cs, srs, scs, iinds, ws, inds, ninds);

	//
	e = es[0];
	tcodes[nodeidx] = tmptcodes[0];

	for(i=1; i<nrands; ++i)
		if(e > es[i])
		{
			e = es[i];
			tcodes[nodeidx] = tmptcodes[i];
		}

	//
	n0 = split_training_data(tcodes[nodeidx], tvals, rs, cs, srs, scs, iinds, ws, inds, ninds);

	//
	grow_subtree(tcodes, lut, 2*nodeidx+1, d+1, maxd, tvals, rs, cs, srs, scs, iinds, ws, &inds[0], n0);
	grow_subtree(tcodes, lut, 2*nodeidx+2, d+1, maxd, tvals, rs, cs, srs, scs, iinds, ws, &inds[n0], ninds-n0);

	//
	return 1;
}

int pico::grow_rtree(int32_t tcodes[], float lut[], int d, float tvals[], int rs[], int cs[], int srs[], int scs[], int iinds[], double ws[], int n)
{
	int i;
	int* inds;

	//
	inds = (int*)malloc(n*sizeof(int));

	for(i=0; i<n; ++i)
		inds[i] = i;

	//
	if(!grow_subtree(tcodes, lut, 0, 0, d, tvals, rs, cs, srs, scs, iinds, ws, inds, n))
	{
		free(inds);
		return 0;
	}
	else
	{
		free(inds);
		return 1;
	}
}

int pico::load_cascade_from_file(const char* path)
{
	int i;
	FILE* file;

	//
	file = fopen(path, "rb");

	if(!file)
		return 0;

	//
	fread(&tsr, sizeof(float), 1, file);
	fread(&tsc, sizeof(float), 1, file);

	fread(&tdepth, sizeof(int), 1, file);

	fread(&ntrees, sizeof(int), 1, file);

	//
	for(i=0; i<ntrees; ++i)
	{
		//
		fread(&tcodes[i][0], sizeof(int32_t), (1<<tdepth)-1, file);
		fread(&luts[i][0], sizeof(float), 1<<tdepth, file);
		fread(&thresholds[i], sizeof(float), 1, file);
	}

	//
	fclose(file);

	//
	return 1;
}

int pico::save_cascade_to_file(const char* path)
{
	int i;
	FILE* file;

	//
	file = fopen(path, "wb");

	if(!file)
		return 0;

	//
	fwrite(&tsr, sizeof(float), 1, file);
	fwrite(&tsc, sizeof(float), 1, file);

	fwrite(&tdepth, sizeof(int), 1, file);

	fwrite(&ntrees, sizeof(int), 1, file);

	//
	for(i=0; i<ntrees; ++i)
	{
		//
		fwrite(&tcodes[i][0], sizeof(int32_t), (1<<tdepth)-1, file);
		fwrite(&luts[i][0], sizeof(float), 1<<tdepth, file);
		fwrite(&thresholds[i], sizeof(float), 1, file);
	}

	//
	fclose(file);

	//
	return 1;
}

float pico::get_tree_output(int i, int r, int c, int sr, int sc, int iind)
{
	int idx, j;

	//
	idx = 1;

	for(j=0; j<tdepth; ++j)
		idx = 2*idx + bintest(tcodes[i][idx-1], r, c, sr, sc, iind);

	//
	return luts[i][idx - (1<<tdepth)];
}

int pico::classify_region(float* o, int r, int c, int s, int iind)
{
	int i, sr, sc;

	//
	if(!ntrees)
		return 1;

	//
	sr = (int)(tsr*s);
	sc = (int)(tsc*s);

	*o = 0.0f;

	//
	for(i=0; i<ntrees; ++i)
	{
		//
		*o += get_tree_output(i, r, c, sr, sc, iind);

		//
		if(*o <= thresholds[i])
			return -1;
	}

	//
	return 1;
}

int pico::learn_new_stage(float mintpr, float maxfpr, int maxntrees, float tvals[], int rs[], int cs[], int ss[], int iinds[], float os[], int np, int nn)
{
	int i;

	int* srs;
	int* scs;

	double* ws;
	double wsum;

	float threshold, tpr, fpr;

	//
	printf("* learning new stage ...\n");

	//
	srs = (int*)malloc((np+nn)*sizeof(int));
	scs = (int*)malloc((np+nn)*sizeof(int));

	for(i=0; i<np+nn; ++i)
	{
		srs[i] = (int)( tsr*ss[i] );
		scs[i] = (int)( tsc*ss[i] );
	}

	//
	ws = (double*)malloc((np+nn)*sizeof(double));

	//
	maxntrees = ntrees + maxntrees;
	fpr = 1.0f;

	while(ntrees<maxntrees && fpr>maxfpr)
	{
		float t;
		int numtps, numfps;

		//
		t = getticks();

		// compute weights ...
		wsum = 0.0;

		for(i=0; i<np+nn; ++i)
		{
			if(tvals[i] > 0)
				ws[i] = exp(-1.0*os[i])/np;
			else
				ws[i] = exp(+1.0*os[i])/nn;

			wsum += ws[i];
		}

		for(i=0; i<np+nn; ++i)
			ws[i] /= wsum;

		// grow a tree ...
		grow_rtree(tcodes[ntrees], luts[ntrees], tdepth, tvals, rs, cs, srs, scs, iinds, ws, np+nn);

		thresholds[ntrees] = -1337.0f;

		++ntrees;

		// update outputs ...
		for(i=0; i<np+nn; ++i)
		{
			float o;

			//
			o = get_tree_output(ntrees-1, rs[i], cs[i], srs[i], scs[i], iinds[i]);

			//
			os[i] += o;
		}

		// get threshold ...
		threshold = 5.0f;

		do
		{
			//
			threshold -= 0.005f;

			numtps = 0;
			numfps = 0;

			//
			for(i=0; i<np+nn; ++i)
			{
				if( tvals[i]>0 && os[i]>threshold)
					++numtps;
				if(	tvals[i]<0 && os[i]>threshold)
					++numfps;
			}

			//
			tpr = numtps/(float)np;
			fpr = numfps/(float)nn;
		}
		while(tpr<mintpr);

		printf("	** tree %d (%d [s]) ... stage tpr=%f, stage fpr=%f\n", ntrees, (int)(getticks()-t), tpr, fpr);
		fflush(stdout);
	}

	//
	thresholds[ntrees-1] = threshold;

	printf("	** threshold set to %f\n", threshold);

	//
	free(srs);
	free(scs);

	free(ws);

	//
	return 1;
}

float pico::sample_training_data(float tvals[], int rs[], int cs[], int ss[], int iinds[], float os[], int* np, int* nn)
{
	int i, n;

	int64_t nw;
	float etpr, efpr;

	int t;

	#define NUMPRNGS 1024
	static int prngsinitialized = 0;
	static uint64_t prngs[NUMPRNGS];

	int stop;

	//
	t = getticks();

	//
	n = 0;

	/*
		object samples
	*/

	for(i=0; i<nobjects; ++i)
		if( classify_region(&os[n], objects[i][0], objects[i][1], objects[i][2], objects[i][3]) == 1 )
		{
			//
			rs[n] = objects[i][0];
			cs[n] = objects[i][1];
			ss[n] = objects[i][2];


			iinds[n] = objects[i][3];

			tvals[n] = +1;

			//
			++n;
		}

	*np = n;

	/*
		non-object samples
	*/

	if(!prngsinitialized)
	{
		// initialize a PRNG for each thread
		for(i=0; i<NUMPRNGS; ++i)
			prngs[i] = 0xFFFF*mwcrand() + 0xFFFF1234FFFF0001LL*mwcrand();

		//
		prngsinitialized = 1;
	}

	//
	nw = 0;
	*nn = 0;

	stop = 0;

	if(nbackground)
	{
		#ifdef OPENMP
			#pragma omp parallel
			{
				int thid;

				//
				thid = omp_get_thread_num();

				while(!stop)
				{
					/*
						data mine hard negatives
					*/

					float o;
					int iind, s, r, c, nrows, ncols;
					uint8_t* pixels;

					//
					iind = background[ mwcrand_r(&prngs[thid])%nbackground ];

					//
					r = mwcrand_r(&prngs[thid])%pdims[iind][0];
					c = mwcrand_r(&prngs[thid])%pdims[iind][1];
					s = objects[mwcrand_r(&prngs[thid])%nobjects][2]; // sample the size of a random object in the pool

					//
					if( classify_region(&o, r, c, s, iind) == 1 )
					{
						//we have a false positive ...
						#pragma omp critical
						{
							if(*nn<*np)
							{
								rs[n] = r;
								cs[n] = c;
								ss[n] = s;

								iinds[n] = iind;

								os[n] = o;

								tvals[n] = -1;

								//
								++n;
								++*nn;
							}
							else
								stop = 1;
						}
					}

					if(!stop)
					{
						#pragma omp atomic
						++nw;
					}
				}
			}
		#endif
	}
	else
		nw = 1;

	/*
		print the estimated true positive and false positive rates
	*/

	etpr = *np/(float)nobjects;
	efpr = (float)( *nn/(double)nw );

	printf("* sampling finished ...\n");
	printf("	** elapsed time: %d\n", (int)(getticks()-t));
	printf("	** cascade TPR=%.8f\n", etpr);
	printf("	** cascade FPR=%.8f (%d/%lld)\n", efpr, *nn, (long long int)nw);

	/*
		
	*/

	return efpr;
}

int pico::learn_with_default_parameters(char* trdata, char* dst)
{
	int i, np, nn;
	float fpr;

	//
	if(!load_training_data(trdata))
	{
		printf("* cannot load training data ...\n");
		return 0;
	}

	//
	tsr = 1.0f;
	tsc = 1.0f;

	tdepth = 5;

	if(!save_cascade_to_file(dst))
			return 0;

	//
	sample_training_data(tvals, rs, cs, ss, iinds, os, &np, &nn);
	learn_new_stage(0.9800f, 0.5f, 4, tvals, rs, cs, ss, iinds, os, np, nn);
	save_cascade_to_file(dst);

	printf("\n");

	sample_training_data(tvals, rs, cs, ss, iinds, os, &np, &nn);
	learn_new_stage(0.9850f, 0.5f, 8, tvals, rs, cs, ss, iinds, os, np, nn);
	save_cascade_to_file(dst);

	printf("\n");

	sample_training_data(tvals, rs, cs, ss, iinds, os, &np, &nn);
	learn_new_stage(0.9900f, 0.5f, 16, tvals, rs, cs, ss, iinds, os, np, nn);
	save_cascade_to_file(dst);

	printf("\n");

	sample_training_data(tvals, rs, cs, ss, iinds, os, &np, &nn);
	learn_new_stage(0.9950f, 0.5f, 32, tvals, rs, cs, ss, iinds, os, np, nn);
	save_cascade_to_file(dst);

	printf("\n");

	//
	while(sample_training_data(tvals, rs, cs, ss, iinds, os, &np, &nn) > 1e-6f)
	{
		learn_new_stage(0.9975f, 0.5f, 64, tvals, rs, cs, ss, iinds, os, np, nn);
		save_cascade_to_file(dst);

		printf("\n");
	}

	//
	printf("* target FPR achieved ... terminating the learning process ...\n");
}


const char* pico::howto()
{
	return
		"TODO\n"
	;
}

int pico::gen_test(int argc, char* argv[])
{
	// initialize the PRNG
	smwcrand(time(0));

	//
	if(argc == 3)
	{
		learn_with_default_parameters(argv[1], argv[2]);
	}
	if(argc == 5)
	{
		sscanf(argv[1], "%f", &tsr);
		sscanf(argv[2], "%f", &tsc);

		sscanf(argv[3], "%d", &tdepth);

		//
		ntrees = 0;

		//
		if(!save_cascade_to_file(argv[4]))
			return 0;

		//
		printf("* initializing: (%f, %f, %d)\n", tsr, tsc, tdepth);

		//
		return 0;
	}
	else if(argc == 7)
	{
		float tpr, fpr;
		int ntrees, np, nn;

		//
		if(!load_cascade_from_file(argv[1]))
		{
			printf("* cannot load a cascade from '%s'\n", argv[1]);
			return 1;
		}

		if(!load_training_data(argv[2]))
		{
			printf("* cannot load the training data from '%s'\n", argv[2]);
			return 1;
		}

		//
		sscanf(argv[3], "%f", &tpr);
		sscanf(argv[4], "%f", &fpr);
		sscanf(argv[5], "%d", &ntrees);

		//
		sample_training_data(tvals, rs, cs, ss, iinds, os, &np, &nn);
		learn_new_stage(tpr, fpr, ntrees, tvals, rs, cs, ss, iinds, os, np, nn);

		//
		if(!save_cascade_to_file(argv[6]))
			return 1;
	}
	else
	{
		printf("%s", howto());
		return 0;
	}

	//
	return 0;
}


void pico::process_webcam_frames(std::string video_path, int & usepyr, void* cascade, float angle, int & minsize, int & maxsize, 
                float & scalefactor, float & stridefactor, float & qthreshold, int & noclustering, int & verbose)
{
	//cv::VideoCapture capture;
    cv::VideoCapture capture(video_path);

	cv::Mat frame;
	cv::Mat framecopy;

	int stop;

	std::string windowname = "--------------------";

	//if(!capture.open(0))
	//{
	//	std::cout << "* cannot initialize video capture ...\n" << std::endl;
	//	return;
	//}
    if (!capture.isOpened())
    {
        printf("Can't open input video file\n");
        return;
    }

	// the main loop
	framecopy = 0;
	stop = 0;

	// framerate
	int64 t1,t0 = cv::getTickCount();
	double fps = 10;

	for(unsigned int fc = 1; fc > 0; fc++)
	{
		// wait 5 miliseconds
		int key = cv::waitKey(1);

		// get the frame from webcam
		//if(!capture.grab())
		//{
		//	stop = 1;
		//	frame = 0;
		//}
		//else
		//	capture.retrieve(frame);
        capture.read(frame);

		// we terminate the loop if the user has pressed 'q'
		if(frame.empty() || key=='q')
			stop = 1;
		else
		{
			// we mustn't tamper with internal OpenCV buffers
			if(framecopy.empty())
				framecopy = cv::Mat(cv::Size(frame.cols, frame.rows), frame.type());
			framecopy = frame.clone();

			// webcam outputs mirrored frames (at least on my machines)
			// you can safely comment out this line if you find it unnecessary
			cv::flip(framecopy, framecopy, 1);

			// ...
			process_image(framecopy, 1, usepyr, cascade, angle, minsize, maxsize, scalefactor, stridefactor, qthreshold, noclustering, verbose);

			// Work out the framerate
			if (fc % 10 == 0 && fc > 0)
			{
				t1 = cv::getTickCount();
				fps = 10.0 / (double(cv::getTickCount()-t0)/cv::getTickFrequency());
				t0 = t1;
			}
			if (fc >= 10)
				std::cout << "FPS: " << fps << std::endl;

			// ...
			cv::imshow(windowname, framecopy);
		}
	}

	// cleanup
	capture.release();
	cv::destroyWindow(windowname);
}

cv::Rect pico::process_image(cv::Mat frame, int draw, int & usepyr, void* cascade, float angle, int & minsize, int & maxsize, 
				float & scalefactor, float & stridefactor, float & qthreshold, int & noclustering, int & verbose)
{
	cv::Rect face_rec;

	int i, j;
	float t;

	uint8_t* pixels;
	int nrows, ncols, ldim;

	#define MAXNDETECTIONS 2048
	int ndetections;
	float qs[MAXNDETECTIONS], rs[MAXNDETECTIONS], cs[MAXNDETECTIONS], ss[MAXNDETECTIONS];

	cv::Mat gray;
	cv::Mat pyr[5];

	/*
		...
	*/

	//
	if(pyr[0].empty())
	{
		//
		gray = cv::Mat(cv::Size(frame.cols, frame.rows), frame.type());

		//
		pyr[0] = gray;
		pyr[1] = cv::Mat(cv::Size(frame.cols/2, frame.rows/2), frame.type());
		pyr[2] = cv::Mat(cv::Size(frame.cols/4, frame.rows/4), frame.type());
		pyr[3] = cv::Mat(cv::Size(frame.cols/8, frame.rows/8), frame.type());
		pyr[4] = cv::Mat(cv::Size(frame.cols/16, frame.rows/16), frame.type());
	}

	// get grayscale image
	if(frame.channels() == 3)
		cv::cvtColor(frame, gray, CV_BGR2GRAY);
	else
		gray = frame.clone();

	// perform detection with the pico library
	t = getticks();

	if(usepyr)
	{
		int nd;

		//
		pyr[0] = gray;

		pixels = (uint8_t*)pyr[0].data;
		nrows = pyr[0].rows;
		ncols = pyr[0].cols;
		ldim = pyr[0].step[0];

		ndetections = find_objects(rs, cs, ss, qs, MAXNDETECTIONS, cascade, angle, pixels, nrows, ncols, ldim, scalefactor, stridefactor, MAX(16, minsize), MIN(128, maxsize));

		for(i=1; i<5; ++i)
		{
			cv::resize(pyr[i-1], pyr[i], pyr[i].size());

			pixels = (uint8_t*)pyr[i].data;
			nrows = pyr[i].rows;
			ncols = pyr[i].cols;
			ldim = pyr[i].step[0];

			nd = find_objects(&rs[ndetections], &cs[ndetections], &ss[ndetections], &qs[ndetections], MAXNDETECTIONS-ndetections, cascade, 0.0f, pixels, nrows, ncols, ldim, scalefactor, stridefactor, MAX(64, minsize>>i), MIN(128, maxsize>>i));

			for(j=ndetections; j<ndetections+nd; ++j)
			{
				rs[j] = (1<<i)*rs[j];
				cs[j] = (1<<i)*cs[j];
				ss[j] = (1<<i)*ss[j];
			}

			ndetections = ndetections + nd;
		}
	}
	else
	{
		//
		pixels = (uint8_t*)gray.data;
		nrows = gray.rows;
		ncols = gray.cols;
		ldim = gray.step[0];

		//
		ndetections = find_objects(rs, cs, ss, qs, MAXNDETECTIONS, cascade, angle, pixels, nrows, ncols, ldim, scalefactor, stridefactor, minsize, MIN(nrows, ncols));
	}

	if(!noclustering)
		ndetections = cluster_detections(rs, cs, ss, qs, ndetections);

	t = getticks() - t;

	for(i=0; i<ndetections; i++)
	{
		if(qs[i]>=qthreshold) // check the confidence threshold
        {
        	face_rec.width = ss[i];
        	face_rec.height = ss[i];
        	face_rec.x = cs[i]-ss[i]/2;
        	face_rec.y = rs[i]-ss[i]/2;
        }
	}

	// if the flag is set, draw each detection
	
	if(draw)
		for(i=0; i<ndetections; ++i)
			if(qs[i]>=qthreshold) // check the confidence threshold
            {
                cv::rectangle(frame, cv::Point(cs[i]-ss[i]/2, rs[i]-ss[i]/2), cv::Point(cs[i]+ss[i]/2, rs[i]+ss[i]/2), cv::Scalar(0,0,255));
				//cv::circle(frame, cv::Point(cs[i], rs[i]), ss[i]/2, cv::Scalar(255, 0, 0), 4, 8, 0); // we draw circles here since height-to-width ratio of the detected face regions is 1.0f
            }
	
	// if the `verbose` flag is set, print the results to standard output
	if(verbose)
	{
		//
		for(i=0; i<ndetections; ++i)
			if(qs[i]>=qthreshold) // check the confidence threshold
				std::cout << (int)rs[i] << " " << (int)cs[i] << " " << (int)ss[i] << " " << qs[i] << std::endl;

		//
		//printf("# %f\n", 1000.0f*t); // use '#' to ignore this line when parsing the output of the program
	}

	//cv::imshow("Pico", frame);

	return face_rec;
}
