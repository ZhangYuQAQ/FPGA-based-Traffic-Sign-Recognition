#include "hls_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ap_fixed.h>

//#include "dscnet_8.h"
#include "dscnet_16.h"



//#define EPSILON	0.1
#define EPSILON 0.1


extern float image[3][56][56];



extern float dw1[3][9];
extern float dw1_tmp[3][3][3];
extern float dw1_bias[3];

extern float pw1[16][3];
extern float pw1_bias[16];

extern float dw2[16][9];
extern float dw2_tmp[16][3][3];
extern float dw2_bias[16];

extern float pw2[32][16];
extern float pw2_bias[32];

extern float dw3[32][9];
extern float dw3_tmp[32][3][3];
extern float dw3_bias[32];

extern float pw3[64][32];
extern float pw3_bias[64];

extern float dw4[64][9];
extern float dw4_tmp[64][3][3];
extern float dw4_bias[64];

extern float pw4[64][64];
extern float pw4_bias[64];

extern float dw5[64][9];
extern float dw5_tmp[64][3][3];
extern float dw5_bias[64];

extern float pw5[64][64];
extern float pw5_bias[64];

extern float fc_weight[62][64];
extern float fc_bias[62];

/////verify HLS part weight
float imge_1_out_PL[3][56][56];
float conv_1_dw_PL[16][16];
float conv_1_dw_bias_PL[16];
float conv_1_pw_PL[16][16];
float conv_1_pw_bias_PL[16];

float image_HLS_out[3][56][56];
float dw_weight_HLS_out[16][16];
float dw1_bias_weight_HLS_out[16];
float pw1_weight_HLS_out[16][16];
float pw1_bias_weight_HLS_out[16];
float pw_weight_HLS_out[16][16];

//////verify HLS conv output
float conv_1_out_PL[3][56][56];
float conv_2_out_PL[16][56][56];
float pool_3_out_PL[16][28][28];

float conv_4_out_PL[16][28][28];
float conv_5_out_PL[32][28][28];
float pool_6_out_PL[32][14][14];

float conv_7_out_PL[32][14][14];
float conv_8_out_PL[64][14][14];
float pool_9_out_PL[64][7][7];

float conv_10_out_PL[64][7][7];
float conv_11_out_PL[64][7][7];

float conv_12_out_PL[64][7][7];
float conv_13_out_PL[64][7][7];
float conv_gap_out_PL[64];
float conv_fc_out_PL[64]; //fc should be 62, hls part is 64

float global_average_pooling_out_PL[64];
float fc_out_PL[62];
//extern FIX_FM DDR_pool_3_out_PL[16][28][28];


///////////////////////////////////c sim
float conv_1_out[3][56][56];
float conv_2_out[16][56][56];
float pool_3_out[16][28][28];

float conv_4_out[16][28][28];
float conv_5_out[32][28][28];
float pool_6_out[32][14][14];

float conv_7_out[32][14][14];
float conv_8_out[64][14][14];
float pool_9_out[64][7][7];

float conv_10_out[64][7][7];
float conv_11_out[64][7][7];

float conv_12_out[64][7][7];
float conv_13_out[64][7][7];

float global_average_pooling_out[64];
float fc_weight_out[62][64];
float fc_input[64];
float conv_13_weight_out[64][64];


///conv weights//
float dw1_tmp_weight[3][3][3];
float dw1_bias_weight[3];
float pw1_weight[16][3];
float pw1_bias_weight[16];

float fc_out[62]; //fc
int sort_out[2];
extern float image_8bit[3][56][56];

////////////////////////////result_conv_nomax[co][h][w]/////////
////////////////////////////result_conv_nomax[co][h][w]//////
float result_conv1_nomax[3][56][56];
float result_conv2_nomax[16][56][56];

float result_conv4_nomax[16][28][28];
float result_conv5_nomax[32][28][28];

float result_conv7_nomax[32][14][14];
float result_conv8_nomax[64][14][14];

using namespace std;

FILE* fo;

float conv_13_weight_PL[64][64];
float conv_12_weight_PL[64][64];
float conv13_bias_out_PL[64];

////////////////////////////golden_model//////////////////////

float max_4(float a1, float a2, float a3, float a4)
{
    float tmp1, tmp2;

    if(a1 > a2) tmp1 = a1; else tmp1 = a2;
    if(a3 > a4) tmp2 = a3; else tmp2 = a4;
    if(tmp1 > tmp2) return tmp1; else return tmp2;
}



void conv_1(
            float input[3][56][56],
            float weight[3][3][3],
            float bias[3],
            float output[3][56][56]
            )
{
    cout << "conv_1..." << endl;

    for(int co = 0; co < 3; co++) {
        for(int h = 0; h < 56; h++) {
            for(int w = 0; w < 56; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {
                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 56 && w+n-1 < 56) ? input[co][h+m-1][w+n-1] : 0);


                    }
                }
                result_conv1_nomax[co][h][w] = sum + bias[co];
                output[co][h][w] = (result_conv1_nomax[co][h][w] > 0)? result_conv1_nomax[co][h][w] : 0.0f;
            }
        }
    }

    fo = fopen("conv_1_out", "w");
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 56; j++) {
            for(int k = 0; k < 56; k ++) {
                fprintf(fo, "conv_1_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

    fo = fopen("dw1_tmp_weight", "w");
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            for(int k = 0; k < 3; k ++) {
                fprintf(fo, "dw1_tmp_weight[%d][%d][%d] = %f\n", i, j, k, weight[i][j][k]);
            }
        }
    }
    fclose(fo);

    fo = fopen("dw1_bias_weight", "w");
    for(int i = 0; i < 3; i++) {
           fprintf(fo, "dw1_bias_weight[%d] = %f\n", i, bias[i]);
    }
    fclose(fo);

    fo = fopen("image_8bit", "w");
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 56; j++) {
            for(int k = 0; k < 56; k ++) {
                fprintf(fo, "image_output[%d][%d][%d] = %f\n", i, j, k, input[i][j][k]);
            }
        }
    }
    fclose(fo);


}


void conv_2(
            float input[3][56][56],
            float weight[16][3],
            float bias[16],
            float output[16][56][56]
            )
{
    cout << "conv_2..." << endl;

    for(int co = 0; co < 16; co++) {
        for(int h = 0; h < 56; h++) {
            for(int w = 0; w < 56; w++) {
                float sum = 0;

                for(int ci = 0; ci < 3; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }
                result_conv2_nomax[co][h][w] = sum + bias[co];
                output[co][h][w] = (result_conv2_nomax[co][h][w] > 0)? result_conv2_nomax[co][h][w] : 0.0f;
            }
        }
    }

    fo = fopen("conv_2_out", "w");
    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 56; j++) {
            for(int k = 0; k < 56; k ++) {
                fprintf(fo, "conv_2_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

    fo = fopen("pw1_weight", "w");
    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 3; j++) {
                fprintf(fo, "pw1_weight[%d][%d] = %f\n", i, j, weight[i][j]);
        }
    }
    fclose(fo);

    fo = fopen("pw1_bias_weight", "w");
    for(int i = 0; i < 16; i++) {
           fprintf(fo, "pw1_bias_weight[%d] = %f\n", i, bias[i]);
    }
    fclose(fo);

}


void max_pool_3(
                   float input[16][56][56],
                   float output[16][28][28]
                   )
{
    cout << "max_pool_3..." << endl;

    for(int co = 0; co < 16; co++) {
        for(int h = 0; h < 28; h++) {
            for(int w = 0; w < 28; w++) {

                output[co][h][w] = max_4(
                                        input[co][h*2][w*2],
                                        input[co][h*2+1][w*2],
                                        input[co][h*2][w*2+1],
                                        input[co][h*2+1][w*2+1]
                                        );
            }
        }
    }

    fo = fopen("max_pool_3_out", "w");
    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 28; j++) {
            for(int k = 0; k < 28; k ++) {
                fprintf(fo, "max_pool_3_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

}


void conv_4(
            float input[16][28][28],
            float weight[16][3][3],
            float bias[16],
            float output[16][28][28]
            )
{
    cout << "conv_4..." << endl;

    for(int co = 0; co < 16; co++) {
        for(int h = 0; h < 28; h++) {
            for(int w = 0; w < 28; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {
                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 28 && w+n-1 < 28) ? input[co][h+m-1][w+n-1] : 0);
                    }
                }
                result_conv4_nomax[co][h][w] = sum + bias[co];
                output[co][h][w] = (result_conv4_nomax[co][h][w] > 0)? result_conv4_nomax[co][h][w] : 0.0f;
            }
        }
    }


    fo = fopen("conv_4_out", "w");
    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 28; j++) {
            for(int k = 0; k < 28; k ++) {
                fprintf(fo, "conv_4_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);



}


void conv_5(
            float input[16][28][28],
            float weight[32][16],
            float bias[32],
            float output[32][28][28]
            )
{
    cout << "conv_5..." << endl;

    for(int co = 0; co < 32; co++) {
        for(int h = 0; h < 28; h++) {
            for(int w = 0; w < 28; w++) {
                float sum = 0;

                for(int ci = 0; ci < 16; ci++ ) {
                	/*if(co==0 && h==0 && w==20)
						printf("%f * %f = %f\n", weight[co][ci], input[ci][h][w], sum);*/

                    sum += weight[co][ci] * input[ci][h][w];
                }
                result_conv5_nomax[co][h][w] = sum + bias[co];
                output[co][h][w] = (result_conv5_nomax[co][h][w] > 0)? result_conv5_nomax[co][h][w] : 0.0f;
            }
        }
    }


    fo = fopen("conv_5_out", "w");
    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 28; j++) {
            for(int k = 0; k < 28; k ++) {
                fprintf(fo, "conv_5_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

	FILE* fo_csim;

	char filename[32];
	sprintf(filename, "pw2_weight");

	fo_csim = fopen(filename, "w");

    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 16; j++) {
                fprintf(fo_csim, "pw2_weight[%d][%d] = %f\n", i, j, weight[i][j]);
        }
    }
    fclose(fo_csim);

    sprintf(filename, "pw2_bias_weight");
    fo_csim = fopen(filename, "w");
    for(int i = 0; i < 32; i++) {
           fprintf(fo_csim, "pw2_bias_weight[%d] = %f\n", i, bias[i]);
    }
    fclose(fo_csim);
}


void max_pool_6(
                float input[32][28][28],
                float output[32][14][14]
                )
{
    cout << "max_pool_6..." << endl;

    for(int co = 0; co < 32; co++) {
        for(int h = 0; h < 14; h++) {
            for(int w = 0; w < 14; w++) {

                output[co][h][w] = max_4(
                                        input[co][h*2][w*2],
                                        input[co][h*2+1][w*2],
                                        input[co][h*2][w*2+1],
                                        input[co][h*2+1][w*2+1]
                                        );
            }
        }
    }


    fo = fopen("max_pool_6_out", "w");
    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 14; j++) {
            for(int k = 0; k < 14; k ++) {
                fprintf(fo, "max_pool_6_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);



}


void conv_7(
            float input[32][14][14],
            float weight[32][3][3],
            float bias[32],
            float output[32][14][14]
            )
{
    cout << "conv_7..." << endl;

    for(int co = 0; co < 32; co++) {
        for(int h = 0; h < 14; h++) {
            for(int w = 0; w < 14; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {

                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 14 && w+n-1 < 14) ? input[co][h+m-1][w+n-1] : 0);
                    }
                }
                float result = sum + bias[co];
                output[co][h][w] = (result > 0)? result : 0.0f;
            }
        }
    }

    fo = fopen("conv_7_out", "w");
    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 14; j++) {
            for(int k = 0; k < 14; k ++) {
                fprintf(fo, "conv_7_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

	FILE* fo_csim;

	char filename[32];
	sprintf(filename, "dw3_weight");

	fo_csim = fopen(filename, "w");

    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 3; j++) {
        	for(int k = 0;k < 3;k++){
                fprintf(fo_csim, "dw3_weight[%d][%d][%d] = %f\n", i, j, k, weight[i][j][k]);
        	}
        }
    }
    fclose(fo_csim);

    sprintf(filename, "dw3_bias_weight");
    fo_csim = fopen(filename, "w");
    for(int i = 0; i < 32; i++) {
           fprintf(fo_csim, "dw3_bias_weight[%d] = %f\n", i, bias[i]);
    }
    fclose(fo_csim);
}


void conv_8(
            float input[32][14][14],
            float weight[64][32],
            float bias[64],
            float output[64][14][14]
            )
{
    cout << "conv_8..." << endl;

    for(int co = 0; co < 64; co++) {
        for(int h = 0; h < 14; h++) {
            for(int w = 0; w < 14; w++) {
                float sum = 0;

                for(int ci = 0; ci < 32; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }

                float result = sum + bias[co];
                output[co][h][w] = (result > 0)? result : 0.0f;
            }
        }
    }

    fo = fopen("conv_8_out", "w");
    for(int i = 0; i < 64; i++) {
        for(int j = 0; j < 14; j++) {
            for(int k = 0; k < 14; k ++) {
                fprintf(fo, "conv_8_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

}


void max_pool_9(
                float input[64][14][14],
                float output[64][7][7]
                )
{
    cout << "max_pool_9..." << endl;

    for(int co = 0; co < 64; co++) {
        for(int h = 0; h < 7; h++) {
            for(int w = 0; w < 7; w++) {

                output[co][h][w] = max_4(
                                        input[co][h*2][w*2],
                                        input[co][h*2+1][w*2],
                                        input[co][h*2][w*2+1],
                                        input[co][h*2+1][w*2+1]
                                        );
            }
        }
    }

    fo = fopen("max_pool_9_out", "w");
    for(int i = 0; i < 64; i++) {
        for(int j = 0; j < 7; j++) {
            for(int k = 0; k < 7; k ++) {
                fprintf(fo, "max_pool_9_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

}



void conv_10(
            float input[64][7][7],
            float weight[64][3][3],
            float bias[64],
            float output[64][7][7]
            )
{
    cout << "conv_10..." << endl;

    for(int co = 0; co < 64; co++) {
        for(int h = 0; h < 7; h++) {
            for(int w = 0; w < 7; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {

                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 7 && w+n-1 < 7) ? input[co][h+m-1][w+n-1] : 0);
                    }
                }
                float result = sum + bias[co];
                output[co][h][w] = (result > 0)? result : 0.0f;
            }
        }
    }

    fo = fopen("conv_10_out", "w");
    for(int i = 0; i < 64; i++) {
        for(int j = 0; j < 7; j++) {
            for(int k = 0; k < 7; k ++) {
                fprintf(fo, "conv_10_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
}


void conv_11(
            float input[64][7][7],
            float weight[64][64],
            float bias[64],
            float output[64][7][7]
            )
{
    cout << "conv_11..." << endl;

    for(int co = 0; co < 64; co++) {
        for(int h = 0; h < 7; h++) {
            for(int w = 0; w < 7; w++) {
                float sum = 0;

                for(int ci = 0; ci < 64; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }
                float result = sum + bias[co];
                output[co][h][w] = (result > 0)? result : 0.0f;
            }
        }
    }

    fo = fopen("conv_11_out", "w");
    for(int i = 0; i < 64; i++) {
        for(int j = 0; j < 7; j++) {
            for(int k = 0; k < 7; k ++) {
                fprintf(fo, "conv_11_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

	FILE* fo_csim;
	char filename[32];
	sprintf(filename, "pw4_weight");

	fo_csim = fopen(filename, "w");

    for(int i = 0; i < 64; i++) {
        for(int j = 0; j < 64; j++) {
            fprintf(fo_csim, "pw4_weight[%d][%d] = %f\n", i, j, weight[i][j]);
        }
    }
    fclose(fo_csim);
}


void conv_12(
            float input[64][7][7],
            float weight[64][3][3],
            float bias[64],
            float output[64][7][7]
            )
{
    cout << "conv_12..." << endl;

    for(int co = 0; co < 64; co++) {
        for(int h = 0; h < 7; h++) {
            for(int w = 0; w < 7; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {

                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 7 && w+n-1 < 7) ? input[co][h+m-1][w+n-1] : 0);
                    }
                }
                float result = sum + bias[co];
                output[co][h][w] = (result > 0)? result : 0.0f;
            }
        }
    }

    fo = fopen("conv_12_out", "w");
    for(int i = 0; i < 64; i++) {
        for(int j = 0; j < 7; j++) {
            for(int k = 0; k < 7; k ++) {
                fprintf(fo, "conv_12_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
}


void conv_13(
            float input[64][7][7],
            float weight[64][64],
            float bias[64],
            float output[64][7][7]
            )
{
    cout << "conv_13..." << endl;

    for(int co = 0; co < 64; co++) {
        for(int h = 0; h < 7; h++) {
            for(int w = 0; w < 7; w++) {
                float sum = 0;

                for(int ci = 0; ci < 64; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }
                float result = sum + bias[co];
                output[co][h][w] = (result > 0)? result : 0.0f;
            }
        }
    }

    fo = fopen("conv_13_weight_out", "w");
    for(int i = 0; i < 64; i++) {
        for(int j = 0; j < 64; j++) {
                fprintf(fo, "conv_13_weight_output[%d][%d] = %f\n", i, j, weight[i][j]);
        }
    }
    fclose(fo);

    fo = fopen("conv_13_out", "w");
    for(int i = 0; i < 64; i++) {
        for(int j = 0; j < 7; j++) {
            for(int k = 0; k < 7; k ++) {
                fprintf(fo, "conv_13_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

	FILE* fo_csim;

	char filename[32];
	sprintf(filename, "conv13_bias_out");
	fo_csim = fopen(filename, "w");

    for(int i = 0; i < 64; i++) {
           fprintf(fo_csim, "cov13_bias_weight[%d] = %f\n", i, bias[i]);
    }
    fclose(fo_csim);
}



void global_average_pooling(float input[64][7][7], float output[64])
{
	cout << "global_average_pooling..." << endl;
	
	float temp[64]={0};
	for (int h=0; h<7; h++){
		for (int w=0; w<7;w++){
			for (int c=0; c<64; c++){
				temp[c]+=input[c][h][w];
			}
		}
	}
	for (int c=0; c<64; c++){
	    output[c]=temp[c]/49;
	
	
	fo = fopen("global_average_pooling_out", "w");
    for(int i = 0; i < 64; i++) {
        fprintf(fo, "global_average_pooling_output[%d] = %f\n", i, output[i]);
    }
    fclose(fo);
}


}

void fc(float input[64], float weights[62][64], float bias[62], float output[62])
{

	cout << "fc..." << endl;
	
	float temp[62]={0};
	for (int i=0; i<62;i++){
		for (int j=0; j<64; j++){
			temp[i]+=weights[i][j]*input[j];//A^T
		}
	}
	for(int i=0; i<62;i++){
		output[i]=temp[i]+bias[i];
	}
	
    fo = fopen("fc_input", "w");
    for(int j = 0; j < 64; j++) {
                fprintf(fo, "fc_input_verify[%d] = %f\n", j,  input[j]);
            }

    fclose(fo);




	fo = fopen("fc_out", "w");
    for(int i = 0; i < 62; i++) {
        fprintf(fo, "fc_output[%d] = %f\n", i, output[i]);
    }
    fclose(fo);

    fo = fopen("fc_weight_out", "w");
    for(int i = 0; i < 62; i++) {
        for(int j = 0; j < 64; j++) {
                fprintf(fo, "fc_weight_output[%d][%d] = %f\n", i, j, weights[i][j]);
            }
    }
    fclose(fo);

}

void sort(float buf1[62],
		  int cla[2]){

	cout << "sort..." << endl;
	float conf_thresh = 0;
    int conf_i = 0;

	for(int i=0; i<62; i++){
		if(buf1[i]>conf_thresh){
			conf_thresh = buf1[i];
			conf_i = i;
		}
	}
	cla[0] = conf_i;
	cla[1] = 0;

	cout << "c_sim result of the picture is :" << cla[0] << endl;
	//printf("traffic sign is :%d\n useless num is :%d\n",cla[0],cla[1]);

	fo = fopen("sort_out", "w");
    for(int i = 0; i < 2; i++) {
        fprintf(fo, "sort_output[%d] = %d\n", i, cla[i]);
    }
    fclose(fo);

}






void golden_model()
{
	conv_1(image, dw1_tmp, dw1_bias, conv_1_out);
	conv_2(conv_1_out, pw1, pw1_bias, conv_2_out);
	max_pool_3(conv_2_out, pool_3_out);

	conv_4(pool_3_out, dw2_tmp, dw2_bias, conv_4_out);
	conv_5(conv_4_out, pw2, pw2_bias, conv_5_out);
	max_pool_6(conv_5_out, pool_6_out);

	conv_7(pool_6_out, dw3_tmp, dw3_bias, conv_7_out);
	conv_8(conv_7_out, pw3, pw3_bias, conv_8_out);
	max_pool_9(conv_8_out, pool_9_out);

	conv_10(pool_9_out, dw4_tmp, dw4_bias, conv_10_out);
	conv_11(conv_10_out, pw4, pw4_bias, conv_11_out);

	conv_12(conv_11_out, dw5_tmp, dw5_bias, conv_12_out);
	conv_13(conv_12_out, pw5, pw5_bias, conv_13_out);

	global_average_pooling(conv_13_out, global_average_pooling_out);
	fc(global_average_pooling_out, fc_weight, fc_bias,fc_out);
	sort(fc_out,sort_out);
}


//////////////////////output_verify//////////////////////
////////////////////////conv16*16*16/////////////////////
void fill_weight_output( int layer, FIX_WT buf[16][16], int ouch, int inch)
{
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 16; j++) {
				switch (layer)
				{
				case 12:
					conv_12_weight_PL[ouch*16+i][inch*16+j] = buf[i][j];
					break;
				case 13:
					conv_13_weight_PL[ouch*16+i][inch*16+j] = buf[i][j];
					break;
				default:
					printf("Wrong layer number.\n");
				}
		}
	}
}
void conv13_weight_HLS_output()
{
	fo = fopen("conv13_weight_HLS_out", "w");
    for(int i = 0; i < 64; i++) {
        for(int j = 0; j < 64; j++) {
              fprintf(fo, "dw_weight_HLS_output[%d][%d] = %f\n", i, j,  conv_13_weight_PL[i][j]);

        }
    }
    fclose(fo);
}

void conv13_bias_HLS_output(FIX_WT buf[64], int ch){
	for(int i=0;i<16;i++){
		conv13_bias_out_PL[ch*16+i] = buf[i];
	}

	FILE* fo_csim;
	char filename[32];
	sprintf(filename, "conv13_bias_out_PL");

	fo_csim = fopen(filename, "w");

    for(int i = 0; i < 64; i++) {
         fprintf(fo_csim, "conv13_bias_out_PL[%d] = %f\n", i, conv13_bias_out_PL[i]);
    }
    fclose(fo_csim);

}

void fill_output_16( int layer, FIX_FM buf[16][16][16], int ch, int col, int row)
{
	for(int i = 0; i < 16; i++) {
		for(int j = 1; j <= 14; j++) {
			for(int k = 1; k <= 14; k++) {
				switch (layer)
				{
				case 1:
					conv_1_out_PL[ch*16+i][col*14+j-1][row*14+k-1] = buf[i][j][k];
					break;
				case 2:
					conv_2_out_PL[ch*16+i][col*14+j-1][row*14+k-1]= buf[i][j][k];
					break;
				case 4:
					conv_4_out_PL[ch*16+i][col*14+j-1][row*14+k-1] = buf[i][j][k];
					break;
				case 5:
					conv_5_out_PL[ch*16+i][col*14+j-1][row*14+k-1] = buf[i][j][k];
					break;
				case 7:
					conv_7_out_PL[ch*16+i][col*14+j-1][row*14+k-1] = buf[i][j][k];
					break;
				case 8:
					conv_8_out_PL[ch*16+i][col*14+j-1][row*14+k-1] = buf[i][j][k];
					break;
				default:
					printf("Wrong layer number.\n");
				}

			}
		}
	}
}




void fill_output_fc(FIX_32_12 buf[16], int ch){
	for(int i=0;i<16;i++){
		conv_fc_out_PL[ch*16+i] = buf[i];
	}

	FILE* fo_csim;
	char filename[32];
	sprintf(filename, "fc_out_PL");

	fo_csim = fopen(filename, "w");

    for(int i = 0; i < 64; i++) {
         fprintf(fo_csim, "fc_out_PL[%d] = %f\n", i, conv_fc_out_PL[i]);
    }
    fclose(fo_csim);

}

void fill_output_gap(FIX_FM buf[16], int ch){
	for(int i=0;i<16;i++){
		conv_gap_out_PL[ch*16+i] = buf[i];
	}

	FILE* fo_csim;
	char filename[32];
	sprintf(filename, "fc_in_PL");

	fo_csim = fopen(filename, "w");

    for(int i = 0; i < 64; i++) {
         fprintf(fo_csim, "fc_in_PL[%d] = %f\n", i, conv_gap_out_PL[i]);
    }
    fclose(fo_csim);
}
///////////////////////////conv16*9*9///////////////////
void fill_output_9( int layer, FIX_FM buf[16][9][9], int ch, int col, int row)
{
	for(int i = 0; i < 16; i++) {
		for(int j = 1; j <= 7; j++) {
			for(int k = 1; k <= 7; k++) {
				switch (layer)
				{
				case 10:
					conv_10_out_PL[ch*16+i][col*7+j-1][row*7+k-1] = buf[i][j][k];
					break;
				case 11:
					conv_11_out_PL[ch*16+i][col*7+j-1][row*7+k-1] = buf[i][j][k];
					break;
				case 12:
					conv_12_out_PL[ch*16+i][col*7+j-1][row*7+k-1] = buf[i][j][k];
					break;
                case 13:
					conv_13_out_PL[ch*16+i][col*7+j-1][row*7+k-1] = buf[i][j][k];
					break;
				default:
					printf("Wrong layer number.\n");
				}

			}
		}
	}
}


void fill_output_pool( int layer, FIX_FM buf[16][16][16], int ch, int col, int row)
{
	for(int i = 0; i < 16; i++) {
		for(int j = 1; j <=14; j++) {
			for(int k = 1; k <=14; k++) {
				switch (layer)
				{
				case 3:
					pool_3_out_PL[ch*16+i][col*14+j-1][row*14+k-1] = buf[i][j][k];
					break;
				case 6:
					pool_6_out_PL[ch*16+i][col*14+j-1][row*14+k-1] = buf[i][j][k];
					break;
				default:
					printf("Wrong layer number.\n");
				}

			}
		}
	}
}

void fill_output_pool9( int layer, FIX_FM buf[16][9][9], int ch, int col, int row)
{
	for(int i = 0; i < 16; i++) {
		for(int j = 1; j <=7; j++) {
			for(int k = 1; k <=7; k++) {
				switch (layer)
				{
				case 9:
					pool_9_out_PL[ch*16+i][col*7+j-1][row*7+k-1] = buf[i][j][k];
					break;
				default:
					printf("Wrong layer number.\n");
				}

			}
		}
	}
}
//////////////////////output_HLS_part_verify//////////////////////
//////////////////////output_HLS_part_verify//////////////////////
void image_HLS_output(FIX_FM buf[16][16][16], int ch, int col, int row)
{
	for(int i = 0; i < 3; i++) {
		for(int j = 1; j <= 14; j++) {
			for(int k = 1; k <= 14; k++) {
				imge_1_out_PL[ch*16+i][col*14+j-1][row*14+k-1] = buf[i][j][k]; //HLS part image
			}
		}
	}
	fo = fopen("image_HLS_out", "w");
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 56; j++) {
            for(int k = 0; k < 56; k ++) {
                fprintf(fo, "image_HLS_output[%d][%d][%d] = %f\n", i, j, k, imge_1_out_PL[i][j][k]);
            }
        }
    }
    fclose(fo);
}



void dw_weight_HLS_output(FIX_WT buf[16][16], int ch)
{
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j <= 15; j++) {
			conv_1_dw_PL[ch*16+i][j] = buf[i][j]; //HLS part image
		}
	}
	fo = fopen("dw_weight_HLS_out", "w");
    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 16; j++) {
              fprintf(fo, "dw_weight_HLS_output[%d][%d] = %f\n", i, j,  conv_1_dw_PL[i][j]);

        }
    }
    fclose(fo);
}

void dw1_bias_weight_HLS_output(FIX_WT buf[16])
{
	for(int i = 0; i < 16; i++) {
			conv_1_dw_bias_PL[i] = buf[i]; //HLS part image
	}
	fo = fopen("dw1_bias_weight_HLS_out", "w");
    for(int i = 0; i < 16; i++) {
              fprintf(fo, "dw1_bias_weight_HLS_out[%d] = %f\n", i,  conv_1_dw_bias_PL[i]);

    }
    fclose(fo);
}

void pw1_weight_HLS_output(FIX_WT buf[16][16], int ch)
{
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j <= 15; j++) {
			conv_1_pw_PL[ch*16+i][j] = buf[i][j]; //HLS part image
		}
	}
	fo = fopen("pw_weight_HLS_out", "w");
    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 16; j++) {
              fprintf(fo, "pw_weight_HLS_output[%d][%d] = %f\n", i, j,  conv_1_pw_PL[i][j]);

        }
    }
    fclose(fo);
}

void pw1_bias_weight_HLS_output(FIX_WT buf[16])
{
	for(int i = 0; i < 16; i++) {
			conv_1_pw_bias_PL[i] = (float)buf[i]; //HLS part image
	}
	fo = fopen("pw1_bias_weight_HLS_out", "w");
    for(int i = 0; i < 16; i++) {
              fprintf(fo, "pw1_bias_weight_HLS_out[%d] = %f\n", i,  conv_1_pw_bias_PL[i]);

    }
    fclose(fo);
}






int PL_golden_compare_layer_1()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_1");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 3; ch++) {
			for(int w = 0; w < 56; w++) {
				for(int h = 0; h < 56; h++) {
				if( abs(conv_1_out_PL[ch][h][w] - conv_1_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_2()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_2");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 16; ch++) {
		for(int w = 0; w < 56; w++) {
			for(int h = 0; h < 56; h++) {
				if( abs(conv_2_out_PL[ch][h][w] - conv_2_out[ch][h][w]) < EPSILON)
				//if( abs(conv_2_out_PL[ch][h][w] - result_conv2_nomax[ch][h][w]) < EPSILON)

				{
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_3()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_3");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 16; ch++) {
			for(int w = 0; w < 28; w++) {
				for(int h = 0; h < 28; h++) {
				if( abs(pool_3_out_PL[ch][h][w] - pool_3_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);


	return pass;
}


int PL_golden_compare_layer_4()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_4");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 16; ch++) {
			for(int w = 0; w < 28; w++) {
				for(int h = 0; h < 28; h++) {
					if( abs(conv_4_out_PL[ch][h][w] - conv_4_out[ch][h][w]) < EPSILON)
					 {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_5()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_5");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 32; ch++) {
			for(int w = 0; w < 28; w++) {
				for(int h = 0; h < 28; h++) {
					if( abs(conv_5_out_PL[ch][h][w] - conv_5_out[ch][h][w]) < EPSILON)
					 {
						fprintf(fo, ".");
					}
					else {
						fprintf(fo, "X");
					}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}



int PL_golden_compare_layer_6()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_6");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 32; ch++) {
			for(int w = 0; w < 14; w++) {
				for(int h = 0; h < 14; h++) {
				if( abs(pool_6_out_PL[ch][h][w] - pool_6_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_7()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_7");


	fo = fopen(filename, "w");

	for(int ch = 0; ch < 32; ch++) {
			for(int w = 0; w < 14; w++) {
				for(int h = 0; h < 14; h++) {
				if( abs(conv_7_out_PL[ch][h][w] - conv_7_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_8()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_8");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 64; ch++) {
			for(int w = 0; w < 14; w++) {
				for(int h = 0; h < 14; h++) {
				if( abs(conv_8_out_PL[ch][h][w] - conv_8_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_9()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_9");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 64; ch++) {
			for(int w = 0; w <7; w++) {
				for(int h = 0; h < 7; h++) {
					if( abs(pool_9_out_PL[ch][h][w] - pool_9_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_10()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_10");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 64; ch++) {
			for(int w = 0; w < 7; w++) {
				for(int h = 0; h < 7; h++) {
				if( abs(conv_10_out_PL[ch][h][w] - conv_10_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_11()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_11");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 64; ch++) {
			for(int w = 0; w < 7; w++) {
				for(int h = 0; h < 7; h++) {
				if( abs(conv_11_out_PL[ch][h][w] - conv_11_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_12()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_12");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 64; ch++) {
			for(int w = 0; w < 7; w++) {
				for(int h = 0; h < 7; h++) {
				if( abs(conv_12_out_PL[ch][h][w] - conv_12_out[ch][h][w]) < EPSILON ) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
    fclose(fo);

	//FILE* fo_csim;
	char filename2[32];
	sprintf(filename2, "conv_12_out_PL");

	fo = fopen(filename2, "w");

    for(int i = 0; i < 64; i++) {
        for(int j = 0; j < 7; j++) {
        	for(int k = 0;k < 7;k++){
                fprintf(fo, "conv_12_out_PL[%d][%d][%d] = %f\n", i, j, k, conv_12_out_PL[i][j][k]);
        	}
        }
    }
    fclose(fo);

	return pass;
}

int PL_golden_compare_layer_13()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_13");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 64; ch++) {
			for(int w = 0; w < 7; w++) {
				for(int h = 0; h < 7; h++) {
					if( abs(conv_13_out_PL[ch][h][w] - conv_13_out[ch][h][w]) < EPSILON )
					 {
						fprintf(fo, ".");
					}
					else {
						fprintf(fo, "X");
					}
				}
				fprintf(fo, "\n");
			}
			fprintf(fo, "\n\n");
	}
    fclose(fo);

	//FILE* fo_csim;
	char filename2[32];
	sprintf(filename2, "conv_13_out_PL");

	fo = fopen(filename2, "w");

    for(int i = 0; i < 64; i++) {
        for(int j = 0; j < 7; j++) {
        	for(int k = 0;k < 7;k++){
                fprintf(fo, "conv_13_out_PL[%d][%d][%d] = %f\n", i, j, k, conv_13_out_PL[i][j][k]);
        	}
        }
    }
    fclose(fo);

	return pass;
}

int PL_golden_compare_layer_gap()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_gap");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 64; ch++) {
		if( abs((float)conv_gap_out_PL[ch] - global_average_pooling_out[ch]) < EPSILON )
		{
			fprintf(fo, ".");
		}
		else {
			fprintf(fo, "X");
		}

			fprintf(fo, "\n");
	}

	return pass;
}

int PL_golden_compare_layer_fc()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_fc");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 62; ch++) {
		if( abs(conv_fc_out_PL[ch] - fc_out[ch]) < EPSILON )
		{
			fprintf(fo, ".");
		}
		else {
			fprintf(fo, "X");
		}

			fprintf(fo, "\n");
	}

	return pass;
}
