
#include "hls_stream.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ap_fixed.h>
#include <stdlib.h>

#include "dscnet_16.h"

using namespace std;

//////////////////
float image[3][56][56];

float dw1[3][3][3];
float dw1_tmp[3][3][3];
float dw1_bias[3];

float pw1[16][3];
float pw1_bias[16];

float dw2[16][9];
float dw2_tmp[16][3][3];
float dw2_bias[16];

float pw2[32][16];
float pw2_bias[32];

float dw3[32][9];
float dw3_tmp[32][3][3];
float dw3_bias[32];

float pw3[64][32];
float pw3_bias[64];

float dw4[64][9];//
float dw4_tmp[64][3][3];
float dw4_bias[64];

float pw4[64][64];
float pw4_bias[64];

float dw5[64][9];
float dw5_tmp[64][3][3];
float dw5_bias[64];

float pw5[64][64];
float pw5_bias[64];

float fc_weight[62][64];
float fc_bias[62];



FIX_WT fix_dw1[16][3][3];
FIX_WT fix_dw1_bias[16];

FIX_WT fix_pw1[16][16];
FIX_WT fix_pw1_bias[16];

FIX_WT  fix_dw2[16][3][3];
FIX_WT  fix_dw2_bias[16];

FIX_WT  fix_pw2[32][16];
FIX_WT  fix_pw2_bias[32];

FIX_WT  fix_dw3[32][3][3];
FIX_WT  fix_dw3_bias[32];

FIX_WT  fix_pw3[64][32];
FIX_WT  fix_pw3_bias[64];

FIX_WT  fix_dw4[64][3][3];
FIX_WT  fix_dw4_bias[64];

FIX_WT  fix_pw4[64][64];
FIX_WT  fix_pw4_bias[64];

FIX_WT  fix_dw5[64][3][3];
FIX_WT  fix_dw5_bias[64];

FIX_WT  fix_pw5[64][64];
FIX_WT  fix_pw5_bias[64];

FIX_WT  fix_fc[64][64];  //62->64
FIX_WT  fix_fc_bias[64];  //62->64

////conv weights///
float image_8bit[3][56][56];
float fix_image_raw_pad_out[3][58][58];

uint16 fix_pw_all_16[59][16][16]; //16 output channel to 128bit
//pw1:0
//pw2:1,2
//pw3:3 4 5 6 7 8 9 10
//pw4: 11-26
//pw5: 27-42
//fc:  43-58

uint16 fix_dw_all_16[12][16][3][3]; //16 output channel to 128bit
//dw1: 0
//dw2: 1
//dw3: 2 3
//dw4: 4 5 6 7
//dw5: 8 9 10 11 

uint16 fix_bias_all_16[31][16]; //16 output channel to 128bit
//0       	:dw1,
//1      	:pw1,
//2			:dw2
//3 4		:pw2
//5 6		:dw3
//7 8 9 10	:pw3
//11 12 13 14:dw4
//15-18		:pw4
//19-22     :dw5
//23-26		:pw5
//27-30     : fc

FIX_FM DDR_pool3_out_PL[16][30][30];
FIX_FM DDR_pool6_out_PL[32][16][16];
FIX_FM DDR_buf[20][16][9][9];

uint256 fix_pw_all_128bit[59][16];
uint256 fix_dw_all_128bit[12][9];
uint256 fix_bias_all_128bit[31];



uint8  fix_image_raw[3][56][56];	// 0~255 RGB raw data
uint8  fix_image_raw_pad[3][58][58];	// 0~255 RGB raw data


uint256 DDR_dw1_pool_out_PL_burst[30*30];		// DDR storage for 1st layer output with padding
uint256 DDR_dw2_pool_out_PL_burst[2*16*16];		// DDR storage for 2nd layer output with padding

//FIX_FM DDR_pool_out_PL[96][82][162];	
uint256 DDR_buf_burst[20*9*9]; // DDR Storage for 3rd layer layers' output

void golden_model();
void reorder_weight_fix();

FILE* fo_i;

int test_one_frame( char* filename )
{

    ///////////// Prepare Image //////////////////////
    std::ifstream ifs_image_raw(filename, std::ios::in | std::ios::binary);
    ifs_image_raw.read((char*)(**fix_image_raw), 3*56*56*sizeof(uint8));  // fix_image_raw[3][56][56]

//    fo_i = fopen("image_8bit", "w");
//    for(int i = 0; i < 3; i++) {
//        for(int j = 0; j < 56; j++) {
//            for(int k = 0; k < 56; k ++) {
//                fprintf(fo_i, "image_output[%d][%d][%d] = %f\n", i, j, k, fix_image_raw[i][j][k]);
//            }
//        }
//    }
//    fclose(fo_i);
    
    std::ifstream ifs_param("traffic_fused_1022.bin", std::ios::in | std::ios::binary); //

    ///////////////// PADDING FOR RAW IMAGE ///////////
    for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 58; j++) {
			for(int k = 0; k < 58; k++) {
				if(j==0 || k==0 || j==57 || k==57) {
					fix_image_raw_pad[i][j][k] = 127; //127 ----- 0
				}
				else {
					fix_image_raw_pad[i][j][k] = fix_image_raw[i][j-1][k-1];
				}
			}
		}
    }

    fo_i = fopen("fix_image_raw_pad_out", "w");

    for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 58; j++) {
			for(int k = 0; k < 58; k++) {
				fprintf(fo_i, "fix_image_raw_pad_out[%d][%d][%d] = %f\n", i, j, k, (float)fix_image_raw_pad[i][j][k]);
				}
			}
		}


    ///////////////// IMAGE NORM ///////////////////
	for(int j = 0; j < 56; j++) {
		for(int k = 0; k < 56; k++) {
			image[0][j][k] = (((fix_image_raw[0][j][k].to_int()/255.0)-0.5)/0.25);
			image[1][j][k] = (((fix_image_raw[1][j][k].to_int()/255.0)-0.5)/0.25);
			image[2][j][k] = (((fix_image_raw[2][j][k].to_int()/255.0)-0.5)/0.25);
		}
	}


    // std::cout << image[0][0][0] << " " << image[1][0][0] << " " << image[2][0][0] << std::endl;

    ///////////// Read Weights ///////////////////////
    ifs_param.read((char*)(*dw1_tmp), 3*3*3*sizeof(float));
    ifs_param.read((char*)dw1_bias, 3*sizeof(float));


    ifs_param.read((char*)(*pw1), 16*3*sizeof(float));
    ifs_param.read((char*)pw1_bias, 16*sizeof(float));

    ifs_param.read((char*)(*dw2_tmp), 16*3*3*sizeof(float));
    ifs_param.read((char*)dw2_bias, 16*sizeof(float));

    ifs_param.read((char*)(*pw2), 32*16*sizeof(float));
    ifs_param.read((char*)pw2_bias, 32*sizeof(float));

    ifs_param.read((char*)(*dw3_tmp), 32*3*3*sizeof(float));
    ifs_param.read((char*)dw3_bias, 32*sizeof(float));

    ifs_param.read((char*)(*pw3), 64*32*sizeof(float));
    ifs_param.read((char*)pw3_bias, 64*sizeof(float));

    ifs_param.read((char*)(*dw4_tmp), 64*3*3*sizeof(float));
    ifs_param.read((char*)dw4_bias, 64*sizeof(float));

    ifs_param.read((char*)(*pw4), 64*64*sizeof(float));
    ifs_param.read((char*)pw4_bias, 64*sizeof(float));

    ifs_param.read((char*)(*dw5_tmp), 64*3*3*sizeof(float));
    ifs_param.read((char*)dw5_bias, 64*sizeof(float));

    ifs_param.read((char*)(*pw5), 64*64*sizeof(float));
    ifs_param.read((char*)pw5_bias, 64*sizeof(float));

    ifs_param.read((char*)(*fc_weight), 62*64*sizeof(float));
    ifs_param.read((char*)fc_bias, 62*sizeof(float));
    std::cout<<"network weights load done"<<std::endl;

    ifs_param.close();


    /////// GOLDEN MODEL ///////////
    printf("Computing Golden Model...\n");
    golden_model();

    reorder_weight_fix();


    int cla[1];


    SEUer(
            fix_image_raw_pad,

            fix_pw_all_128bit,
            fix_dw_all_128bit,
            fix_bias_all_128bit,

            DDR_dw1_pool_out_PL_burst,
            DDR_dw2_pool_out_PL_burst,
            DDR_buf_burst,
            //int *debug_pin,
            cla
    );
    //printf(" HLS traffic sign is :%d\n useless num is :%d\n",cla[0],cla[1]);
    printf(" printf HLS traffic sign is :%d\n",cla[0]);
    std::cout<<" cout HLS traffic sign is :"<<dec<<cla[0]<<"\n"<<std::endl;

    return 0;
}


int main()
{


	printf("parking_45_3.bin\n");
	char filename[32];

	sprintf(filename, "stop1_1020_P_tranpose1.bin");

	//sprintf(filename, "62_1_1022_P_tranpose1.bin");

	//sprintf(filename, "39_2_tranpose1.bin");

	//sprintf(filename, "park_45_2_tranpose1.bin");

	//test_one_frame("62_1_1022_P_tranpose1.bin");



	test_one_frame(filename);

	//char filename[20];
	//sprintf(filename, "stop1_1020_P_tranpose1.bin");

	//test_one_frame("62_1_1022_P_tranpose1.bin");

	//test_one_frame("heng22_1_1022_P_tranpose1.bin");

	//test_one_frame("park_45_2_tranpose1.bin");
	//test_one_frame("stop1_1020_P_tranpose1.bin");
	//test_one_frame("39_1_tranpose1.bin");
	//test_one_frame("39_2_tranpose1.bin");
	//test_one_frame("39_2_astype1.bin");




	return 0;

}


