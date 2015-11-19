
#include "pico.h"

void* cascade = 0;

int minsize;
int maxsize;

float angle;

float scalefactor;
float stridefactor;

float qthreshold;

int usepyr;
int noclustering;
int verbose;

using namespace FacePico;

int main(int argc, char* argv[])
{
	//
	int arg;
	char input[1024], output[1024];

	FacePico::pico * pico = new FacePico::pico();

	//
	if(argc<2 || 0==strcmp("-h", argv[1]) || 0==strcmp("--help", argv[1]))
	{
		printf("Usage: pico <path/to/cascade> <options>...\n");
		printf("Detect objects in images.\n");
		printf("\n");

		// command line options
		printf("Mandatory arguments to long options are mandatory for short options too.\n");
		printf("  -i,  --input=PATH          set the path to the input image\n");
		printf("                               (*.jpg, *.png, etc.)\n");
		printf("  -o,  --output=PATH         set the path to the output image\n");
		printf("                               (*.jpg, *.png, etc.)\n");
		printf("  -m,  --minsize=SIZE        sets the minimum size (in pixels) of an\n");
		printf("                               object(default is 128)\n");
		printf("  -M,  --maxsize=SIZE        sets the maximum size (in pixels) of an\n");
		printf("                               object(default is 1024)\n");
		printf("  -q,  --qthreshold=THRESH   detection quality threshold (>=0.0):\n");
		printf("                               all detections with estimated quality\n");
		printf("                               below this threshold will be discarded\n");
		printf("                               (default is 5.0)\n");
		printf("  -c,  --scalefactor=SCALE   how much to rescale the window during the\n");
		printf("                               multiscale detection process (default is 1.1)\n");
		printf("  -t,  --stridefactor=STRIDE how much to move the window between neighboring\n");
		printf("                               detections (default is 0.1, i.e., 10 percent)\n");
		printf("  -u,  --usepyr              turns on the coarse image pyramid support\n");
		printf("  -n,  --noclustering        turns off detection clustering\n");
		printf("  -v,  --verbose             print details of the detection process\n");
		printf("                               to `stdout`\n");

		//
		printf("Exit status:\n");
		printf(" 0 if OK,\n");
		printf(" 1 if trouble (e.g., invalid path to input image).\n");

		//
		return 0;
	}
	else
	{
		int size;
		FILE* file;

		//
		//file = fopen(argv[1], "rb");
		file = fopen("../../cascades/facefinder", "rb");

		if(!file)
		{
			printf("# cannot read cascade from '%s'\n", argv[1]);
			return 1;
		}

		//
		fseek(file, 0L, SEEK_END);
		size = ftell(file);
		fseek(file, 0L, SEEK_SET);

		//
		cascade = malloc(size);

		if(!cascade || size!=fread(cascade, 1, size, file))
			return 1;

		//
		fclose(file);
	}

	// set default parameters
	minsize = 128;
	maxsize = 1024;

	scalefactor = 1.1f;
	stridefactor = 0.1f;

	qthreshold = 5.0f;

	usepyr = 1;
	noclustering = 0;
	verbose = 1;

	//
	input[0] = 0;
	output[0] = 0;

	// parse command line arguments
	arg = 2;

	while(arg < argc)
	{
		//
		if(0==strcmp("-u", argv[arg]) || 0==strcmp("--usepyr", argv[arg]))
		{
			usepyr = 1;
			++arg;
		}
		else if(0==strcmp("-i", argv[arg]) || 0==strcmp("--input", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				sscanf(argv[arg+1], "%s", input);
				arg = arg + 2;
			}
			else
			{
				printf("# missing argument after '%s'\n", argv[arg]);
				return 1;
			}
		}
		else if(0==strcmp("-o", argv[arg]) || 0==strcmp("--output", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				sscanf(argv[arg+1], "%s", output);
				arg = arg + 2;
			}
			else
			{
				printf("# missing argument after '%s'\n", argv[arg]);
				return 1;
			}
		}
		else if(0==strcmp("-m", argv[arg]) || 0==strcmp("--minsize", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				sscanf(argv[arg+1], "%d", &minsize);
				arg = arg + 2;
			}
			else
			{
				printf("# missing argument after '%s'\n", argv[arg]);
				return 1;
			}
		}
		else if(0==strcmp("-M", argv[arg]) || 0==strcmp("--maxsize", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				sscanf(argv[arg+1], "%d", &maxsize);
				arg = arg + 2;
			}
			else
			{
				printf("# missing argument after '%s'\n", argv[arg]);
				return 1;
			}
		}
		else if(0==strcmp("-c", argv[arg]) || 0==strcmp("--scalefactor", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				sscanf(argv[arg+1], "%f", &scalefactor);
				arg = arg + 2;
			}
			else
			{
				printf("# missing argument after '%s'\n", argv[arg]);
				return 1;
			}
		}
		else if(0==strcmp("-t", argv[arg]) || 0==strcmp("--stridefactor", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				sscanf(argv[arg+1], "%f", &stridefactor);
				arg = arg + 2;
			}
			else
			{
				printf("# missing argument after '%s'\n", argv[arg]);
				return 1;
			}
		}
		else if(0==strcmp("-q", argv[arg]) || 0==strcmp("--qthreshold", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				sscanf(argv[arg+1], "%f", &qthreshold);
				arg = arg + 2;
			}
			else
			{
				printf("# missing argument after '%s'\n", argv[arg]);
				return 1;
			}
		}
		else if(0==strcmp("-n", argv[arg]) || 0==strcmp("--noclustering", argv[arg]))
		{
			noclustering = 1;
			++arg;
		}
		else if(0==strcmp("-v", argv[arg]) || 0==strcmp("--verbose", argv[arg]))
		{
			verbose = 1;
			++arg;
		}
		else
		{
			printf("# invalid command line argument '%s'\n", argv[arg]);
			return 1;
		}
	}

	if(verbose)
	{
		//
		printf("# Copyright (c) 2013, Nenad Markus\n");
		printf("# All rights reserved.\n\n");

		printf("# cascade parameters:\n");
		printf("#	tsr = %f\n", ((float*)cascade)[0]);
		printf("#	tsc = %f\n", ((float*)cascade)[1]);
		printf("#	tdepth = %d\n", ((int*)cascade)[2]);
		printf("#	ntrees = %d\n", ((int*)cascade)[3]);
		printf("# detection parameters:\n");
		printf("#	minsize = %d\n", minsize);
		printf("#	maxsize = %d\n", maxsize);
		printf("#	scalefactor = %f\n", scalefactor);
		printf("#	stridefactor = %f\n", stridefactor);
		printf("#	qthreshold = %f\n", qthreshold);
		printf("#	usepyr = %d\n", usepyr);
	}

	//
	if(0 == input[0])
		pico->process_webcam_frames("/home/jxgu/dataset/save/pose_reco.avi", usepyr, cascade, minsize, maxsize, 
                scalefactor, stridefactor, qthreshold, noclustering, verbose);
	else
	{
		cv::Mat img;

		//
		img = cv::imread(input, -1);
		if(img.empty())
		{
			std::cout << "# cannot load image" << std::endl;
			return 1;
		}

		int64 t0 = cv::getTickCount();
		pico->process_image(img, 1, usepyr, cascade, minsize, maxsize, scalefactor, stridefactor, qthreshold, noclustering, verbose);
		std::cout << "Time elapsed: " << (double(cv::getTickCount()-t0)/cv::getTickFrequency()) << " seconds" << std::endl;

		//
		if(0!=output[0])
			cv::imwrite(output, img);
		else if(!verbose)
		{
			cv::imwrite(input, img);
			cv::waitKey(0);
		}

	}

	return 0;
}
