#include <cstddef>
#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

//#define CSIM_DEBUG

#define FM_RG			11
#define FM_ACC_RG		11
#define WT_RG			11

#define WID 16 //14+2
#define HEI 16
#define CHA 16

#define WID_9 9
#define HEI_9 9

	typedef ap_fixed<12,  4, AP_RND, AP_SAT> FIX_FM;	//fix point for feature map
	typedef ap_fixed<12,  4, AP_RND, AP_SAT> FIX_FM_acc;	//fix point for accumulation
	typedef ap_fixed<12,  4, AP_RND, AP_SAT> FIX_WT;	//fix point for weights

	typedef ap_fixed<16, 8, AP_RND, AP_SAT> FIX_16_8;
	typedef ap_fixed<16, 6, AP_RND, AP_SAT> FIX_16_6;
	typedef ap_fixed<16, 5, AP_RND, AP_SAT> FIX_16_5;
	typedef ap_fixed<16, 4, AP_RND, AP_SAT> FIX_16_4;
	typedef ap_fixed<16, 3, AP_RND, AP_SAT> FIX_16_3;
	typedef ap_fixed<16, 10, AP_RND, AP_SAT> FIX_16_10;
	typedef ap_fixed<32,16, AP_RND, AP_SAT> FIX_32_16;
	typedef ap_fixed<32,12, AP_RND, AP_SAT> FIX_32_12;
	typedef ap_fixed<32,10, AP_RND, AP_SAT> FIX_32_10;
	typedef ap_fixed<32, 4, AP_RND, AP_SAT> FIX_32_4;
	typedef ap_fixed<32, 7, AP_RND, AP_SAT> FIX_32_7;
	typedef ap_fixed<32,25, AP_RND, AP_SAT> FIX_32_25;

	typedef ap_uint<1> uint1;
	typedef ap_uint<2> uint2;
	typedef ap_uint<4> uint4;
	typedef ap_uint<8> uint8;
	typedef ap_uint<16> uint16;

	typedef ap_uint<128> uint128;
	typedef ap_uint<256> uint256;


void Conv2D(
		FIX_FM in7[CHA][9][9],
		FIX_FM out7[CHA][9][9],
		FIX_FM in[CHA][HEI][WID],
		FIX_FM out[CHA][HEI][WID],
		FIX_WT weights[CHA][CHA],
		FIX_WT bias[CHA],
		uint2 mode, uint1 relu); //mode = 0 3x3depthwise; mode = 1 1x1 conv2d

void SEUer(

				// uint32 image_oringinal[58*58],
				uint8 image_oringinal[3][58][58],
				//uint8 image_oringinal[3*58*58],

				uint256 conv_weight_1x1_all[59][16],
				uint256 conv_weight_3x3_all[12][9],
				// uint128 conv_weight_3x3_all[16][9],
				uint256 bias_all[31],


				uint256 DDR_dw1_pool_out_PL_burst[16/16*30*30],
				uint256 DDR_dw2_pool_out_PL_burst[32/16*16*16],
				uint256 DDR_buf_burst[20*9*9],
				int cla[1]
);

#ifdef CSIM_DEBUG
void fill_output_16( int layer, FIX_FM buf[16][16][16], int ch, int col, int row);
void fill_output_9( int layer, FIX_FM buf[16][9][9], int ch, int col, int row);
void fill_output_pool( int layer, FIX_FM buf[16][16][16], int ch, int col, int row);
void fill_output_pool9( int layer, FIX_FM buf[16][9][9], int ch, int col, int row);

void dw_weight_HLS_output(FIX_WT buf[16][16], int ch);
void image_HLS_output(FIX_FM buf[16][16][16], int ch, int col, int row);
void dw1_bias_weight_HLS_output(FIX_WT buf[16]);
void pw1_bias_weight_HLS_output(FIX_WT buf[16]);
void pw1_weight_HLS_output(FIX_WT buf[16][16], int ch);
void fill_output_fc(FIX_32_12 buf[16], int ch);
void fill_output_gap(FIX_FM buf[16], int ch);


void fill_weight_output( int layer, FIX_WT buf[16][16], int inch, int ouch);
void conv13_weight_HLS_output();
void conv13_bias_HLS_output(FIX_WT buf[64], int ch);

int PL_golden_compare_layer_1();
int PL_golden_compare_layer_2();
int PL_golden_compare_layer_3();
int PL_golden_compare_layer_4();
int PL_golden_compare_layer_5();
int PL_golden_compare_layer_6();
int PL_golden_compare_layer_7();
int PL_golden_compare_layer_8();
int PL_golden_compare_layer_9();
int PL_golden_compare_layer_10();
int PL_golden_compare_layer_11();
int PL_golden_compare_layer_12();
int PL_golden_compare_layer_13();

int PL_golden_compare_layer_gap();

int PL_golden_compare_layer_fc();
#endif



