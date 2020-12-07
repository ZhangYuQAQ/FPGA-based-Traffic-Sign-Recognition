#include "hls_stream.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>

//#include "dscnet_8.h"
#include "dscnet_16.h"

#include <ap_fixed.h>

using namespace std;

/* floating point input weights */

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
FIX_WT zero_wt=0;


/* fixed point parameters */
extern FIX_WT fix_dw1[16][3][3];  //3->16
extern FIX_WT fix_dw1_bias[16];

extern FIX_WT fix_pw1[16][16];  //3->16
extern FIX_WT fix_pw1_bias[16];

extern FIX_WT  fix_dw2[16][3][3];
extern FIX_WT  fix_dw2_bias[16];

extern FIX_WT  fix_pw2[32][16];
extern FIX_WT  fix_pw2_bias[32];

extern FIX_WT  fix_dw3[32][3][3];
extern FIX_WT  fix_dw3_bias[32];

extern FIX_WT  fix_pw3[64][32];
extern FIX_WT  fix_pw3_bias[64];

extern FIX_WT  fix_dw4[64][3][3];
extern FIX_WT  fix_dw4_bias[64];

extern FIX_WT  fix_pw4[64][64];
extern FIX_WT  fix_pw4_bias[64];

extern FIX_WT  fix_dw5[64][3][3];
extern FIX_WT  fix_dw5_bias[64];

extern FIX_WT  fix_pw5[64][64];
extern FIX_WT  fix_pw5_bias[64];

extern FIX_WT  fix_fc[64][64];  //62->64
extern FIX_WT  fix_fc_bias[64];  //62->64

FILE* fo1;



FIX_WT fix_pw_all[59][16][16]; //16 output channel to 128bit
//pw1:0
//pw2:1,2
//pw3:3 4 5 6 7 8 9 10
//pw4: 11-26
//pw5: 27-42
//fc:  43-58

FIX_WT fix_dw_all[12][16][3][3]; //16 output channel to 128bit
//dw1: 0
//dw2: 1
//dw3: 2 3
//dw4: 4 5 6 7
//dw5: 8 9 10 11 

FIX_WT fix_bias_all[31][16]; //16 output channel to 128bit
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
extern uint16 fix_pw_all_16[59][16][16]; //16 output channel to 128bit

extern uint16 fix_dw_all_16[12][16][3][3]; //16 output channel to 128bit

extern uint16 fix_bias_all_16[31][16]; //16 output channel to 128bit

extern uint256 fix_pw_all_128bit[59][16];
extern uint256 fix_dw_all_128bit[12][9];
extern uint256 fix_bias_all_128bit[31];

/////reorder weight
float dw1_bias_reorder[16];
float pw1_bias_reorder[16];
float fix_dw1_bias_f[16];
float fix_pw1_bias_f[16];
float fix_bias_all_f[31][16];

void reorder_weight_fix()
{
	// for dw1
	for(int m = 0; m < 3; m++) {
		for(int n = 0; n < 3; n++) {
			for(int c = 0; c < 16; c++) {
				if(c < 3) {
					fix_dw1[c][m][n] = (FIX_WT)dw1_tmp[c][m][n];
					fix_dw1_bias[c] = (FIX_WT)dw1_bias[c];
					fix_dw1_bias_f[c] = (float)fix_dw1_bias[c];
					//std::cout<<"\n fix_dw1_bias c<3: "<<fix_dw1_bias[c]<<std::endl;
				}
				else {
					fix_dw1[c][m][n] = 0;
					fix_dw1_bias[c] = 0;
					fix_dw1_bias_f[c] = 0;
					//std::cout<<"\n fix_dw1_bias c>3: "<<fix_dw1_bias[c]<<std::endl;
				}
			}
		}
	}

	fo1 = fopen("dw1_bias_reorder", "w");
	for(int i = 0; i < 16; i++) {
			  fprintf(fo1, "dw1_bias_reorder[%d] = %f\n", i,  fix_dw1_bias_f[i]);
	}
	fclose(fo1);


	//for pw1
	for(int co = 0; co < 16; co++) {
		fix_pw1_bias[co] = (FIX_WT)pw1_bias[co];
		fix_pw1_bias_f[co] = (float)fix_pw1_bias[co];
		for(int ci = 0; ci < 16; ci++) {
			if(ci < 3) {
				fix_pw1[co][ci] = (FIX_WT)pw1[co][ci];
			}
			else{
				fix_pw1[co][ci] = (FIX_WT)0;
			}
		}
	}

	fo1 = fopen("pw1_bias_reorder", "w");
	for(int i = 0; i < 16; i++) {
			  fprintf(fo1, "pw1_bias_reorder[%d] = %f\n", i,  fix_pw1_bias_f[i]);
	}
	fclose(fo1);
////////////////////////////////////////////////////////////////////////////////////////
	//for dw2
	for(int c = 0; c < 16; c++) {
		fix_dw2_bias[c] = (FIX_WT)dw2_bias[c];
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++) {
				fix_dw2[c][m][n] = (FIX_WT)dw2_tmp[c][m][n];
			}
		}
	}

	//for pw2
	for(int co = 0; co < 32; co++) {
		fix_pw2_bias[co] = (FIX_WT)pw2_bias[co];
		for(int ci = 0; ci < 16; ci++) {
			fix_pw2[co][ci] = (FIX_WT)pw2[co][ci]; //pw2[32][16]
		}
	}
/////////////////////////////////////////////////////////////////////////////////////////////
//for dw3
	for(int c = 0; c < 32; c++) {
		fix_dw3_bias[c] = (FIX_WT)dw3_bias[c];
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++) {
				fix_dw3[c][m][n] = (FIX_WT)dw3_tmp[c][m][n];
			}
		}
	}

	//for pw3
	for(int co = 0; co < 64; co++) {
		fix_pw3_bias[co] = (FIX_WT)pw3_bias[co];
		for(int ci = 0; ci < 32; ci++) {
			fix_pw3[co][ci] = (FIX_WT)pw3[co][ci]; //pw3[64][32]
		}
	}
//////////////////////////////////////////////////////////////////////////////////////////////////
	//for dw4
	for(int c = 0; c < 64; c++) {
		fix_dw4_bias[c] = (FIX_WT)dw4_bias[c];
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++) {
				fix_dw4[c][m][n] = (FIX_WT)dw4_tmp[c][m][n];
			}
		}
	}

	//for pw4
	for(int co = 0; co < 64; co++) {
		fix_pw4_bias[co] = (FIX_WT)pw4_bias[co];
		for(int ci = 0; ci < 64; ci++) {
			fix_pw4[co][ci] = (FIX_WT)pw4[co][ci]; //pw4[64][64]
		}
	}
///////////////////////////////////////////////////////////////////////////////////////////////
//for dw5
	for(int c = 0; c < 64; c++) {
		fix_dw5_bias[c] = (FIX_WT)dw5_bias[c];
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++) {
				fix_dw5[c][m][n] = (FIX_WT)dw5_tmp[c][m][n];
			}
		}
	}

	//for pw5
	for(int co = 0; co < 64; co++) {
		fix_pw5_bias[co] = (FIX_WT)pw5_bias[co];
		for(int ci = 0; ci < 64; ci++) {
			fix_pw5[co][ci] = (FIX_WT)pw5[co][ci]; //pw5[64][64]
		}
	}

//////////////////////////////////////////////////////////////////////////////////////////////////////
///for fc bias
	for(int co = 0; co < 64; co++) {
		if(co<62){
			fix_fc_bias[co] = (FIX_WT)fc_bias[co];
		}
		else{
			fix_fc_bias[co] = zero_wt;
		}
	}

	for(int ci = 0; ci < 64; ci++) {
		for(int co = 0; co < 64; co++) {
			if (co<62) {
				fix_fc[co][ci] = (FIX_WT)fc_weight[co][ci]; //pw2[32][16]
			}
			else{
				fix_fc[co][ci] = zero_wt;
			}
		}
	}

//////////// reorder conv_1x1 weights, and put all weights together


	int index_3x3 = -1;
	int index_1x1 = -1;
	int index_bias = -1;
	int CO_N, CI_N;

	// dw1_conv_3x3 weights and bias/////////////////////////////////////////////////////////////////
	for(int c = 0; c < 16; c++) {
		if( c % 16 == 0) {
			index_3x3++;
			index_bias++;
		}
		fix_bias_all[index_bias][c] = fix_dw1_bias[c];//fix_bias_all[0]=dw1 bias

		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++)
				fix_dw_all[index_3x3][c][m][n] = fix_dw1[c][m][n]; //fix_dw_all[0]=dw1 weight

		}
	}
//	std::cout<<"\n dw1 index_bias: "<<index_bias<<std::endl;
//	for(int m = 0; m < 1; m++) {
//		for(int n = 0; n < 16; n++) {
//			std::cout<<"\n dw1 fix_bias_all: "<<fix_bias_all[m][n]<<std::endl;
//		}
//	}

//index_3x3=0
//index_bias=0
	// pw1_conv_1x1 weights (reorder) and bias////////////////////////////////////////
	CO_N = 16 / 16;
	CI_N = 16 / 16;
	for(int CO = 0; CO < CO_N; CO++) {
		for(int CI = 0; CI < CI_N; CI++) {
			index_1x1++;

			for(int co = 0; co < 16; co++) {
				for(int ci = 0; ci < 16; ci++) {
					fix_pw_all[index_1x1][co][ci] = fix_pw1[co + CO * 16][ci + CI * 16];
					///FIX_WT fix_pw1[16][16]; fix_pw_all[0]=pw1 weight
				}
			}
		}
	}
	for(int CO = 0; CO < CO_N; CO++) {
		index_bias++;
		for(int co = 0; co < 16; co++) {
			fix_bias_all[index_bias][co] = fix_pw1_bias[co + CO * 16];//fix_bias_all[1]=pw1 bias
		}
	}


	std::cout<<"\n pw1 index_bias: "<<index_bias<<std::endl;
//	 index_3x3 = 0;
//	 index_1x1 = 0;
//	 index_bias = 1;


	// dw2_conv_3x3 weights and bias/////////////////////////////////////////////////////////////////
	for(int c = 0; c < 16; c++) {
		if( c % 16 == 0) {
			index_3x3++;
			index_bias++;
		}
		fix_bias_all[index_bias][c] = fix_dw2_bias[c];//fix_bias_all[2]=dw2 bias
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++)
				fix_dw_all[index_3x3][c][m][n] = fix_dw2[c][m][n]; //fix_dw_all[1]=dw2 weight
		}
	}
	std::cout<<"\n dw2 index_bias: "<<index_bias<<std::endl;

	//	 index_3x3 = 1;
	//	 index_1x1 = 0;
	//	 index_bias = 2;
/////////////////////////////////////	pw2_conv_1x1 weights (reorder) and bias//////////////////////////

	CO_N = 32 / 16;
	CI_N = 16 / 16;
	for(int CO = 0; CO < CO_N; CO++) {
		for(int CI = 0; CI < CI_N; CI++) {
			index_1x1++;

			for(int co = 0; co < 16; co++) {
				for(int ci = 0; ci < 16; ci++) {
					fix_pw_all[index_1x1][co][ci] = fix_pw2[co + CO * 16][ci + CI * 16];
					///FIX_WT fix_pw2[32][16]; fix_pw_all[1,2]=pw2 [0-15][16], [16-31][16]weight
				}
			}
		}
	}
	for(int CO = 0; CO < CO_N; CO++) {
		index_bias++;
		for(int co = 0; co < 16; co++) {
			fix_bias_all[index_bias][co] = fix_pw2_bias[co + CO * 16];//fix_bias_all[3,4]=pw2 bias
		}
	}
	std::cout<<"\n pw2 index_bias: "<<index_bias<<std::endl;

	//	 index_3x3 = 1;
	//	 index_1x1 = 2;
	//	 index_bias = 4;

///////////////////////////////////////dw3////////////////////////////////////////////////
	for(int c = 0; c < 32; c++) {
		if( c % 16 == 0) {
			index_3x3++;
			index_bias++;
		}
		fix_bias_all[index_bias][c%16] = fix_dw3_bias[c];//fix_bias_all[5,6]=dw3 bias
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++)
				fix_dw_all[index_3x3][c%16][m][n] = fix_dw3[c][m][n]; //fix_dw_all[2,3]=dw3 weight
		}
	}
	std::cout<<"\n dw3 index_bias: "<<index_bias<<std::endl;

	//	 index_3x3 = 3;
	//	 index_1x1 = 2;
	//	 index_bias = 6;
///////////////////////////////////////pw3///////////////////////////////////////////////////
	CO_N = 64 / 16;
	CI_N = 32 / 16;
	for(int CO = 0; CO < CO_N; CO++) {//4
		for(int CI = 0; CI < CI_N; CI++) {//2
			index_1x1++;

			for(int co = 0; co < 16; co++) {
				for(int ci = 0; ci < 16; ci++) {
					fix_pw_all[index_1x1][co][ci] = fix_pw3[co + CO * 16][ci + CI * 16];
					///FIX_WT fix_pw3[64][32]; fix_pw_all[3-10]=pw3 [0-15][0-15],[0-15][16-31]....weight
				}
			}
		}
	}
	for(int CO = 0; CO < CO_N; CO++) {
		index_bias++;
		for(int co = 0; co < 16; co++) {
			fix_bias_all[index_bias][co] = fix_pw3_bias[co + CO * 16];//fix_bias_all[7,8.9,10]=pw3 bias
		}
	}
	//	 index_3x3 = 3;
	//	 index_1x1 = 10;
	//	 index_bias = 10;
//////////////////////////////////////////////////////dw4///////////////////////////////////
	for(int c = 0; c < 64; c++) {
		if( c % 16 == 0) {
			index_3x3++;
			index_bias++;
		}
		fix_bias_all[index_bias][c%16] = fix_dw4_bias[c];//fix_bias_all[11,12,13,14]=dw4 bias
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++)
				fix_dw_all[index_3x3][c%16][m][n] = fix_dw4[c][m][n]; //fix_dw_all[4,5,6,7]=dw4 weight
		}
	}
	//	 index_3x3 = 7;
	//	 index_1x1 = 10;
	//	 index_bias = 14;


//////////////////////////////////////////////pw4//////////////////////////////////////////////////
	CO_N = 64 / 16;
	CI_N = 64 / 16;
	for(int CO = 0; CO < CO_N; CO++) {//4
		for(int CI = 0; CI < CI_N; CI++) {//4
			index_1x1++;

			for(int co = 0; co < 16; co++) {
				for(int ci = 0; ci < 16; ci++) {
					fix_pw_all[index_1x1][co][ci] = fix_pw4[co + CO * 16][ci + CI * 16];
					///FIX_WT fix_pw4[64][64]; fix_pw_all[11-26]=pw4 [0-15][0-15],[0-15][16-31],[0-15][32-47]....weight
				}
			}
		}
	}
	for(int CO = 0; CO < CO_N; CO++) {
		index_bias++;
		for(int co = 0; co < 16; co++) {
			fix_bias_all[index_bias][co] = fix_pw4_bias[co + CO * 16];//fix_bias_all[15,16,17,18]=pw3 bias
		}
	}
	std::cout<<"\n pw4 index_bias: "<<index_bias<<std::endl;

	//	 index_3x3 = 7;
	//	 index_1x1 = 26;
	//	 index_bias = 18;


//////////////////////////////////////////////////////dw5///////////////////////////////////
	for(int c = 0; c < 64; c++) {
		if( c % 16 == 0) {
			index_3x3++;
			index_bias++;
		}
		fix_bias_all[index_bias][c%16] = fix_dw5_bias[c];//fix_bias_all[19-22]=dw5 bias
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++)
				fix_dw_all[index_3x3][c%16][m][n] = fix_dw5[c][m][n]; //fix_dw_all[8-11]=dw5 weight
		}
	}
	std::cout<<"\n dw5 index_bias: "<<index_bias<<std::endl;

	//	 index_3x3 = 11;
	//	 index_1x1 = 26;
	//	 index_bias = 22;
//////////////////////////////////////////////pw5//////////////////////////////////////////////////
	CO_N = 64 / 16;
	CI_N = 64 / 16;
	for(int CO = 0; CO < CO_N; CO++) {//4
		for(int CI = 0; CI < CI_N; CI++) {//4
			index_1x1++;

			for(int co = 0; co < 16; co++) {
				for(int ci = 0; ci < 16; ci++) {
					fix_pw_all[index_1x1][co][ci] = fix_pw5[co + CO * 16][ci + CI * 16];
					///FIX_WT fix_pw5[64][64]; fix_pw_all[27-42]=pw5 [0-15][0-15],[0-15][16-31],[0-15][32-47]....weight
				}
			}
		}
	}
	for(int CO = 0; CO < CO_N; CO++) {
		index_bias++;
		for(int co = 0; co < 16; co++) {
			fix_bias_all[index_bias][co] = fix_pw5_bias[co + CO * 16];//fix_bias_all[22-26]=pw5 bias
		}
	}
	std::cout<<"\n pw5 index_bias: "<<index_bias<<std::endl;
	std::cout<<"\n pw5 index_index_1x1: "<<index_1x1<<std::endl;

	//	 index_3x3 = 11;
	//	 index_1x1 = 42;
	//	 index_bias = 26;
///////////////////////////////////////////////fc layer ///////////////////////////////////////////////////////////
	CO_N = 64 / 16;
	CI_N = 64 / 16;
	for(int CO = 0; CO < CO_N; CO++) {//4
		for(int CI = 0; CI < CI_N; CI++) {//4
			index_1x1++;

			for(int co = 0; co < 16; co++) {
				for(int ci = 0; ci < 16; ci++) {
					fix_pw_all[index_1x1][co][ci] = fix_fc[co + CO * 16][ci + CI * 16];
					///FIX_WT fix_fc[64][64]; fix_pw_all[43-58]=fc [0-15][0-15],[0-15][16-31],[0-15][32-47]....weight
				}
			}
		}
	}
	for(int CO = 0; CO < CO_N; CO++) {
		index_bias++;
		for(int co = 0; co < 16; co++) {
			fix_bias_all[index_bias][co] = fix_fc_bias[co + CO * 16];//fix_bias_all[27 28 29 30]=pw5 bias
		}
	}

//	for(int i=0;i<16;i++){
//		for(int j=0;j<16;j++){
//			std::cout<<" fix_pw_all[57] i:"<<dec<<(i)<<" j:"<<dec<<j<<" :"<<fix_pw_all[57][i][j]<<"  "<<std::endl;
//		}
//		std::cout<<"\n"<<std::endl;
//	}
//	std::cout<<"\n"<<std::endl;
//
//	for(int i=0;i<16;i++){
//		std::cout<<"\n dw3 bias:   "<<dec<<(16+i)<<"    :"<<fix_bias_all[30][i]<<std::endl;
//	}

	std::cout<<"\n fc index_bias: "<<index_bias<<std::endl;
	std::cout<<"\n fc index_1x1: "<<index_1x1<<std::endl;

	//	 index_3x3 = 11;
	//	 index_1x1 = 58;
	//	 index_bias = 30;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
	for(int i = 0; i < 139; i++) {
		for(int m = 0; m < 16; m++) {
			for(int n = 0; n < 16; n++) {
				uint8 DATA = 0;
				DATA.range(7, 0) = fix_pw_all[i][m][n].range(7, 0);
				fix_pw_all8[i][m][n].range(7, 0) = DATA.range(7, 0);
			}
		}
	}


	for(int i = 0; i < 16; i++) {
		for(int k = 0; k < 32; k++) {
			for(int m = 0; m < 3; m++) {
				for(int n = 0; n < 3; n++) {
					uint8 DATA = 0;
					DATA.range(7, 0) = fix_dw_all[i][k][m][n].range(7, 0);
					fix_dw_all8[i][k][m][n].range(7, 0) = DATA.range(7, 0);
				}
			}
		}
	}

	for(int i = 0; i < 43; i++) {
		for(int k = 0; k < 32; k++) {
			uint8 DATA = 0;
			DATA.range(7, 0) = fix_bias_all[i][k].range(7, 0);
			fix_bias_all8[i][k].range(7, 0) = DATA.range(7, 0);
		}
	}
*/

	index_3x3++;
	index_1x1++;
	index_bias++;

	///////////////////////////////////////// write weights_fixed.bin////////////////////////////
	std::ofstream ofs_param_write_128("traffic_fused_reorder.bin", std::ios::out | std::ios::binary);
	for(int i = 0; i < index_1x1; i++) {
		for(int k = 0; k < 16; k++) {

			uint256 DATA = 0;
			for(int j = 0; j < 16; j ++) {
				DATA.range(j*16 + WT_RG, j*16) = fix_pw_all[i][j][k].range(WT_RG, 0);
			}
			fix_pw_all_128bit[i][k].range(255, 0) = DATA.range(255, 0);
		}
	}
	ofs_param_write_128.write((char*)fix_pw_all_128bit, index_1x1 * 16 * sizeof(uint256));

	// fill fix_conv_weight_3x3_all into 256 bit-width bus
	for(int i = 0; i < index_3x3; i++) {
		for(int m = 0; m < 3; m++) {
			for(int n = 0; n < 3; n++) {

				uint256 DATA = 0;
				for(int j = 0; j < 16; j++) {
					DATA.range(j*16 + WT_RG, j*16) = fix_dw_all[i][j][m][n].range(WT_RG, 0);
				}
				fix_dw_all_128bit[i][m*3+n].range(255, 0) = DATA.range(255, 0);
			}
		}
	}
	ofs_param_write_128.write((char*)fix_dw_all_128bit, index_3x3 * 3 * 3 * sizeof(uint256));

	// fill fix_bias_all into 128 bit-width bus
	//verify  bias order


//	fo1 = fopen("fix_bias_all_f_reorder", "w");
//	for(int i = 0; i < 16; i++) {
//			  fprintf(fo1, "fix_bias_all_f_reorder[%d] = %f\n", i,  fix_dw1_bias_f[i]);
//	}
//	fclose(fo1);

	for(int i = 0; i < index_bias; i++) {
		uint256 DATA = 0;
		for(int j = 0; j < 16; j++) {
			DATA.range(j*16 + WT_RG, j*16) = fix_bias_all[i][j].range(WT_RG, 0);
		}
		fix_bias_all_128bit[i].range(255, 0) = DATA.range(255, 0);
	}
	std::cout<<"\n fix_bias_all_128bit: "<<hex<<fix_bias_all_128bit[0]<<std::endl;
//	for(int m = 0; m < 1; m++) {
//			for(int n = 0; n < 16; n++) {
//				std::cout<<"\n fc dw1 fix_bias_all: "<<hex<<fix_bias_all[m][n]<<std::endl;
//			}
//		}
	ofs_param_write_128.write((char*)fix_bias_all_128bit, index_bias * sizeof(uint256));
	std::cout<<"write done"<<std::endl;
	ofs_param_write_128.close();

}
