//#include "dscnet_8.h"
#include "dscnet_16.h"

#include <math.h>
#include <fstream>
#include <hls_math.h>
#include <ap_fixed.h>
#include <string.h>

using namespace std;

// feature map buffers  16*16*16
FIX_FM FM_buf1[CHA][HEI][WID]; //16 16 16
FIX_FM FM_buf2[CHA][HEI][WID];
FIX_FM FM_buf3[CHA][HEI][WID];
FIX_FM FM_buf4[CHA][HEI][WID];
FIX_FM FM_buf5[CHA][HEI][WID];

//FIX_FM_acc FM_buf_acc[CHA][HEI][WID];
FIX_FM FM_buf_acc[CHA][HEI][WID];

// feature map buffers  16*9*9
FIX_FM FM_buf10[CHA][HEI_9][WID_9];
FIX_FM FM_buf11[CHA][HEI_9][WID_9];
FIX_FM FM_buf12[CHA][HEI_9][WID_9];
FIX_FM FM_buf13[CHA][HEI_9][WID_9];
FIX_FM FM_buf14[CHA][HEI_9][WID_9];



FIX_FM FC_IN[4][16];
//FIX_FM FC_IN1[16];
//FIX_FM FC_IN2[16];
//FIX_FM FC_IN3[16];
//FIX_FM FC_IN4[16];
// FIX_FM FC_OUT[62];

// weight buffers
FIX_WT weight_buf[4][CHA][CHA];

FIX_WT bias_buf[4][16];

FIX_WT weight_buf1[CHA][CHA];
FIX_WT weight_buf2[CHA][CHA];
FIX_WT weight_buf3[CHA][CHA];
FIX_WT weight_buf4[CHA][CHA];

FIX_16_8 bias_buf16_8[4][16];
FIX_32_12 fc_buf[4][16];


//FIX_WT weight_buf_1x1[4][16][16];
//FIX_WT weight_buf_3x3[4][16][3][3];
//FIX_FM FM_buf_pool[16][7][7];

void FC(FIX_FM in[16],
	  FIX_WT w[16][16],
	  FIX_32_12 out[16])
{
	#pragma HLS array_partition variable=in dim=1 complete
	#pragma HLS array_partition variable=w dim=2 complete
	#pragma HLS array_partition variable=out dim=1 complete

	//FIX_WT tmp1[16];
	FIX_32_12 tmp2;

	#pragma HLS array_partition variable=tmp1 dim=1 complete
	for (int i=0; i<16;i++){
//#pragma HLS PIPELINE II=1
		for (int j=0; j<16; j++){
#pragma HLS unroll
			tmp2=out[i]+in[j]*w[i][j];
			out[i]=tmp2;
		}
	}
}

void gap_fc(FIX_FM bias_buf[16],
	  FIX_32_12 fc_buf[16])
{
	#pragma HLS array_partition variable=bias_buf dim=1 complete
	#pragma HLS array_partition variable=fc_buf dim=1 complete
		for (int j=0; j<16; j++){
#pragma HLS unroll
			fc_buf[j]=bias_buf[j];

		}
}

void sort(FIX_32_12 buf1[16],
		FIX_32_12 buf2[16],
		FIX_32_12 buf3[16],
		FIX_32_12 buf4[16],
		int cla[1]){

	FIX_32_12 conf_thresh = -100.0;

#ifdef CSIM_DEBUG
std::cout<<"inital conf_thresh:"<<conf_thresh<<std::endl;
std::cout<<"\n"<<std::endl;
#endif

    int conf_i = 0;
    //int zero=0;
	for(int i=0; i<16; i++){
		if(buf1[i]>conf_thresh){
			conf_thresh = buf1[i];
			conf_i = i;
		}
	}

#ifdef CSIM_DEBUG  //print feature map
std::cout<<"1-16 conf_thresh:"<<conf_thresh<<std::endl;
std::cout<<"\n"<<std::endl;
#endif

	for(int i=0; i<16; i++){
		if(buf2[i]>conf_thresh){
			conf_thresh = buf2[i];
			conf_i = i + 16;
		}
	}

#ifdef CSIM_DEBUG  //print feature map
std::cout<<"17-32 conf_thresh:"<<conf_thresh<<std::endl;
std::cout<<"\n"<<std::endl;
#endif

	for(int i=0; i<16; i++){
		if(buf3[i]>conf_thresh){
			conf_thresh = buf3[i];
			conf_i = i + 32;
		}
	}
#ifdef CSIM_DEBUG  //print feature map
std::cout<<"33-48 conf_thresh:"<<conf_thresh<<std::endl;
std::cout<<"\n"<<std::endl;
#endif

	for(int i=0; i<14; i++){
		if(buf4[i]>conf_thresh){
			conf_thresh = buf4[i];
			conf_i = i + 48;
		}
	}

#ifdef CSIM_DEBUG  //print feature map
std::cout<<"49-62 conf_thresh:"<<conf_thresh<<std::endl;
std::cout<<"\n"<<std::endl;
#endif

	cla[0] = conf_i;
	//cla[1] = zero;
}

//after dw 3*3.size:16*9*9, all buf copy to ddr
void relu_copy_buf_to_DDR( uint256* dest, int buf_id, FIX_FM src[16][HEI_9][WID_9])
{
	uint256* dest_ptr = dest + 9*9*buf_id;

	for(int h = 0; h < HEI_9; h++) {
		for(int w = 0; w < WID_9; w++) {
#pragma HLS pipeline

			uint256 DATA = 0;
			for(int c = 0; c < 16; c++) {
#pragma HLS unroll
				FIX_FM d = src[c][h][w]; //do the func need?
				DATA.range(FM_RG + 16*c, 16*c) = d.range(FM_RG, 0);
			}
			dest_ptr[w].range(255, 0) = DATA.range(255, 0);
		}
		dest_ptr += 9;
	}
}



//after pw 1*1,size:16*9*9,all buf 16*9*9 copy to ddr
void relu_copy_buf_to_DDR_acc( uint256* dest, int buf_id, FIX_FM src[16][9][9])
{
	uint256* dest_ptr = dest + 9 * 9* buf_id;

	for(int h = 0; h < 9; h++) {

		for(int w = 0; w < 9; w++) {
#pragma HLS pipeline
			uint256 DATA = 0;
			for(int c = 0; c < 16; c++) {
#pragma HLS unroll
				FIX_FM d = src[c][h][w];
				DATA.range(FM_RG + 16*c, 16*c) = d.range(FM_RG, 0);
			}
			dest_ptr[w].range(255, 0) = DATA.range(255, 0);
		}
		dest_ptr += 9;
	}
}




FIX_FM img_norm_ch[256] = {
		-2.000000, -1.984314, -1.968627, -1.952941, -1.937255, -1.921569, -1.905882, -1.890196, -1.874510, -1.858824, -1.843137, -1.827451, -1.811765, -1.796078, -1.780392, -1.764706, -1.749020,
		-1.733333, -1.717647, -1.701961, -1.686275, -1.670588, -1.654902, -1.639216, -1.623529, -1.607843, -1.592157, -1.576471, -1.560784, -1.545098, -1.529412, -1.513725, -1.498039,
		-1.482353, -1.466667, -1.450980, -1.435294, -1.419608, -1.403922, -1.388235, -1.372549, -1.356863, -1.341176, -1.325490, -1.309804, -1.294118, -1.278431, -1.262745, -1.247059,
		-1.231373, -1.215686, -1.200000, -1.184314, -1.168627, -1.152941, -1.137255, -1.121569, -1.105882, -1.090196, -1.074510, -1.058824, -1.043137, -1.027451, -1.011765, -0.996078,
		-0.980392, -0.964706, -0.949020, -0.933333, -0.917647, -0.901961, -0.886275, -0.870588, -0.854902, -0.839216, -0.823529, -0.807843, -0.792157, -0.776471, -0.760784, -0.745098,
		-0.729412, -0.713725, -0.698039, -0.682353, -0.666667, -0.650980, -0.635294, -0.619608, -0.603922, -0.588235, -0.572549, -0.556863, -0.541176, -0.525490, -0.509804, -0.494118,
		-0.478431, -0.462745, -0.447059, -0.431373, -0.415686, -0.400000, -0.384314, -0.368627, -0.352941, -0.337255, -0.321569, -0.305882, -0.290196, -0.274510, -0.258824, -0.243137,
		-0.227451, -0.211765, -0.196078, -0.180392, -0.164706, -0.149020, -0.133333, -0.117647, -0.101961, -0.086275, -0.070588, -0.054902, -0.039216, -0.023529, -0.007843, 0.007843,
		0.023529, 0.039216, 0.054902, 0.070588, 0.086275, 0.101961, 0.117647, 0.133333, 0.149020, 0.164706, 0.180392, 0.196078, 0.211765, 0.227451, 0.243137, 0.258824,
		0.274510, 0.290196, 0.305882, 0.321569, 0.337255, 0.352941, 0.368627, 0.384314, 0.400000, 0.415686, 0.431373, 0.447059, 0.462745, 0.478431, 0.494118, 0.509804,
		0.525490, 0.541176, 0.556863, 0.572549, 0.588235, 0.603922, 0.619608, 0.635294, 0.650980, 0.666667, 0.682353, 0.698039, 0.713725, 0.729412, 0.745098, 0.760784,
		0.776471, 0.792157, 0.807843, 0.823529, 0.839216, 0.854902, 0.870588, 0.886275, 0.901961, 0.917647, 0.933333, 0.949020, 0.964706, 0.980392, 0.996078, 1.011765,
		1.027451, 1.043137, 1.058824, 1.074510, 1.090196, 1.105882, 1.121569, 1.137255, 1.152941, 1.168627, 1.184314, 1.200000, 1.215686, 1.231373, 1.247059, 1.262745,
		1.278431, 1.294118, 1.309804, 1.325490, 1.341176, 1.356863, 1.372549, 1.388235, 1.403922, 1.419608, 1.435294, 1.450980, 1.466667, 1.482353, 1.498039, 1.513725,
		1.529412, 1.545098, 1.560784, 1.576471, 1.592157, 1.607843, 1.623529, 1.639216, 1.654902, 1.670588, 1.686275, 1.701961, 1.717647, 1.733333, 1.749020, 1.764706,
		1.780392, 1.796078, 1.811765, 1.827451, 1.843137, 1.858824, 1.874510, 1.890196, 1.905882, 1.921569, 1.937255, 1.952941, 1.968627, 1.984314, 2.000000
};

void load_image_normal(FIX_FM img_buf[CHA][HEI][WID], uint8 image_in_raw_pad_burst[3][58][58],
							int col, int row)
{
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 16; j++) {
#ifdef CSIM_DEBUG
			if(i + col*14 == 0 || i + col*14 == 57 || j + row*14 == 0 || j + row*14 == 57 )
				img_buf[0][i][j] = 0.0;
			else
#endif
			img_buf[0][i][j] = img_norm_ch[(image_in_raw_pad_burst[0][i + col*14][j + row*14]).to_uint()];
		}
	}

	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 16; j++) {
#ifdef CSIM_DEBUG
			if(i + col*14 == 0 || i + col*14 == 57 || j + row*14 == 0 || j + row*14 == 57 )
				img_buf[1][i][j] = 0.0;
			else
#endif
			img_buf[1][i][j] = img_norm_ch[(image_in_raw_pad_burst[1][i + col*14][j + row*14]).to_uint()];
		}
	}

	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 16; j++) {
#ifdef CSIM_DEBUG
			if(i + col*14 == 0 || i + col*14 == 57 || j + row*14 == 0 || j + row*14 == 57 )
				img_buf[2][i][j] = 0.0;
			else
#endif
			img_buf[2][i][j] = img_norm_ch[(image_in_raw_pad_burst[2][i + col*14][j + row*14]).to_uint()];
		}
	}
}

void load_weight_3D_from_axi( FIX_WT dest[16][3][3], uint256 src[9])
{
	for(int m = 0; m < 3; m++) {
		for(int n = 0; n < 3; n++) {
#pragma HLS pipeline
			uint256 DATA = 0;
			DATA.range(255, 0) = src[3*m+n].range(255, 0);
			for(int c = 0; c < CHA; c++) {
#pragma HLS unroll

				dest[c][m][n].range(WT_RG, 0) = DATA.range(WT_RG + c*16, c*16);

			}
		}
	}
}

void load_weight_3x3_from_axi( FIX_WT dest[CHA][CHA], uint256 src[9])  // weight3*3 [16][9],instead of [16][3][3]
{

	for(int n = 0; n < 9; n++) {
#pragma HLS pipeline
		uint256 DATA = 0;
		DATA.range(255, 0) = src[n].range(255, 0);
		for(int c = 0; c < CHA; c++) {
#pragma HLS unroll

			dest[c][n].range(WT_RG, 0) = DATA.range(WT_RG + c*16, c*16);

		}
	}

}

void load_weight_1x1_from_axi( FIX_WT dest[CHA][CHA], uint256 src[16])  //128/8=16 FOR 16 OUTPUT CHANNEL
{

	for(int ci = 0; ci < CHA; ci++) {
#pragma HLS pipeline
		uint256 DATA = 0;
		DATA.range(255, 0) = src[ci].range(255, 0);
		for(int co = 0; co < 16; co++) {
#pragma HLS unroll
			dest[co][ci].range(WT_RG, 0) = DATA.range(WT_RG + co*16, co*16);
		}
	}
}

void load_bias_from_axi(FIX_WT dest[16], uint256 src)
{
	uint256 DATA = 0;
	DATA.range(255, 0) = src.range(255, 0);
	for(int c = 0; c < 16; c++) {
#pragma HLS unroll

		dest[c].range(WT_RG, 0) = DATA.range(WT_RG + c*16, c*16);
	}
}

void load_bias_from_axi_fc(FIX_16_8 dest[16], uint256 src)
{
	uint256 DATA = 0;
	DATA.range(255, 0) = src.range(255, 0);
	for(int c = 0; c < 16; c++) {
#pragma HLS unroll

		dest[c].range(15, 0) = DATA.range(15 + c*16, c*16);
	}
}

void set_bias(FIX_FM buf[CHA][HEI][WID], FIX_WT bias[CHA])
{
#pragma HLS array_partition variable=buf dim=1 complete
#pragma HLS array_partition variable=bias dim=1 complete
	for(int h = 1; h <= HEI-2; h+=2) {
		for(int w = 1; w <= WID-2; w++) {
#pragma HLS pipeline
			for(int c = 0; c < CHA; c++) {
#pragma HLS unroll
				buf[c][h  ][w] = bias[c]; //h=13
				buf[c][h+1][w] = bias[c]; //h=14
			}
		}
	}
}



void Relu( FIX_FM buf[16][9][9] )
{
	for(int j = 0; j <9; j++) {
		for(int k = 0; k < 9; k++) {
#pragma HLS pipeline
			for(int i = 0; i < 16; i++) {
#pragma HLS unroll
				if( buf[i][j][k] < 0 ) {
					buf[i][j][k] = 0;
				}
			}
		}
	}
}

void set_bias7(FIX_FM buf[CHA][HEI_9][WID_9], FIX_WT bias[16])
{

#pragma HLS array_partition variable=buf dim=1 complete
#pragma HLS array_partition variable=bias dim=1 complete
	for(int h = 1; h <= 7; h++) {
		for(int w = 1; w <= 7; w++) {
#pragma HLS pipeline
			for(int c = 0; c < 16; c++) {
#pragma HLS unroll
				buf[c][h][w] = bias[c];
			}
		}
	}
}

//dw2 begin,load_dw1_pool results.16*28*28��//ch=0
void load_dw1_pool_from_DDR( uint256* ddr_dw1_pool_burst,
							 FIX_FM buf[CHA][HEI][WID],
							 int ch, int col, int row)
{
	uint256* ddr_dw1_pool_burst_ptr =ddr_dw1_pool_burst + ch*30*30 + col*14*30 + row*14;

	for(int h = 0; h < 16; h++) {
		for(int w = 0; w < 16; w++) {
#pragma HLS pipeline
			uint256 DATA = 0;
			DATA.range(255, 0) = ddr_dw1_pool_burst_ptr[w].range(255, 0);//w:1-15
			for(int c = 0; c < 16 ; c++) {
#pragma HLS unroll

				buf[c][h][w].range(FM_RG, 0) = DATA.range(FM_RG + c*16, c*16);
			}
		}
		ddr_dw1_pool_burst_ptr += 30;
	}
}


//dw2 begin,load_dw1_pool results,//col=0,  row=0
void load_dw2_pool_from_DDR( uint256* ddr_dw2_pool_burst,
							 FIX_FM buf[CHA][HEI][WID],
							 int ch, int col, int row)
{
	uint256* ddr_dw2_pool_burst_ptr = ddr_dw2_pool_burst + ch*16*16 + col*14*16 + row*14;

	for(int h = 0; h < 16; h++) {
		for(int w = 0; w < 16; w++) {
#pragma HLS pipeline
			uint256 DATA = 0;
			DATA.range(255, 0) = ddr_dw2_pool_burst_ptr[w].range(255, 0);
			for(int c = 0; c < 16; c++) {
#pragma HLS unroll

				buf[c][h][w].range(FM_RG, 0) = DATA.range(FM_RG + c*16, c*16);
			}
		}
		ddr_dw2_pool_burst_ptr += 16;
	}
}

void load_buf_from_DDR( FIX_FM dest[16][HEI_9][WID_9], uint256* src, int buf_id)
{
	uint256* src_ptr = src + 9*9*buf_id;

	for(int h = 0; h < HEI_9; h++) {

		for(int w = 0; w < WID_9; w++) {
#pragma HLS pipeline II=1

			uint256 DATA = src_ptr[w];
			for(int c = 0; c < 16; c++) {
#pragma HLS unroll
				dest[c][h][w].range(FM_RG, 0) = DATA.range(FM_RG + c*16, c*16);
			}
		}
		src_ptr += 9;
	}
}

/////////Don't do max pool on hls side
inline FIX_FM relu_single( FIX_FM d ) {
	if( d < 0 )
		return 0;
	else
		return d;
}


inline FIX_FM relu_max(FIX_FM a, FIX_FM b, FIX_FM c, FIX_FM d)
{
	FIX_FM t1, t2;

	if(a > b) t1 = relu_single(a);
	else t1 = relu_single(b);

	if(c > d) t2 = relu_single(c);
	else t2 = relu_single(d);

	if(t1 > t2) return t1;
	else return t2;
}

inline FIX_FM max(FIX_FM a, FIX_FM b, FIX_FM c, FIX_FM d)
{
	FIX_FM t1, t2;

	if(a > b) t1 = a;
	else t1 = b;

	if(c > d) t2 = c;
	else t2 = d;

	if(t1 > t2) return t1;
	else return t2;
}


void Relu_Max_Pooling(
		 FIX_FM  buf_in[CHA][HEI][WID],
		 uint256* ddr_dw1_pool_burst,
		 uint256* ddr_dw2_pool_burst,
		 uint256* ddr_buf, int buf_id,
		 int ch, int col, int row, int layer)//dw1_burst:ch=0; dw2_burst:col.row=-;
{

	uint256* buf_in_ptr = ddr_buf + buf_id*9*9 + 9;
	uint256* ddr_dw1_pool_burst_ptr = ddr_dw1_pool_burst + ch*30*30 + (1 + col*7)*30 + (row*7);
	uint256* ddr_dw2_pool_burst_ptr = ddr_dw2_pool_burst + ch*16*16 + (1 + col*7)*16 + (row*7); //(1 + col*7)*14 padding


#pragma HLS array_partition variable=buf_in dim=1 complete

	for(int h = 1; h <= 7; h++) {
		for(int w = 1; w <= 7; w++) {
#pragma HLS pipeline II=2
			uint256 DATA = 0;
			for(int c = 0; c < 16; c++) {
#pragma HLS unroll
				FIX_FM d = relu_max(buf_in[c][h*2-1][w*2-1], buf_in[c][h*2-1][w*2],
							   buf_in[c][h*2][w*2-1], buf_in[c][h*2][w*2]);

				DATA.range(FM_RG + c*16, c*16) = d.range(FM_RG, 0);
			}

			if( layer == 1 ) {
				ddr_dw1_pool_burst_ptr[w].range(255, 0) = DATA.range(255, 0);
			}
			else if( layer == 2 ) {
				ddr_dw2_pool_burst_ptr[w].range(255, 0) = DATA.range(255, 0);
			}
			else if( layer == 3 ) {
				buf_in_ptr[w].range(255, 0) = DATA.range(255, 0);
			}
		}

		buf_in_ptr += 9;
		ddr_dw1_pool_burst_ptr += 30;
		ddr_dw2_pool_burst_ptr += 16;
	}
}


void global_pooling(FIX_FM buf[CHA][9][9], FIX_FM fc_in[16])
{
#pragma HLS array_partition variable=buf dim=1 complete

	FIX_32_25 temp[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},tmp2;
#pragma HLS array_partition variable=temp dim=1 complete

	for (int h=1; h<8; h++){
		for (int w=1; w<8;w++){
#pragma HLS pipeline
			for (int c=0; c<CHA; c++){
#pragma HLS unroll
				temp[c]+=buf[c][h][w];
			}
		}
	}
	for (int c=0; c<CHA; c++){
#pragma HLS pipeline
	    //fc_in[c]=(FIX_FM)temp[c]/49;
		//fc_in[c]=temp[c]/49;
		//tmp2=temp[c]/49;
		fc_in[c]=temp[c]/49;

//#ifdef CSIM_DEBUG
//		std::cout<<" c:"<<dec<<c<<" "<<" gap tmp2:"<<tmp2<<std::endl;
//		std::cout<<"\n"<<std::endl;
//#endif


	}
}

void clear_buffer( FIX_FM buf[CHA][HEI_9][WID_9] )
{
	FIX_FM fix_zero=0;
	for(int h = 0; h < HEI_9; h++) {
		for(int w = 0; w < WID_9; w++) {
#pragma HLS pipeline
			for(int c = 0; c < CHA; c++) {
#pragma HLS unroll
				buf[c][h][w] = fix_zero;
			}
		}
	}
}

void SEUer(

				// uint32 image_oringinal[58*58],
				//uint8 image_oringinal[3*58*58],
				uint8 image_oringinal[3][58][58],

				uint256 conv_weight_1x1_all[59][16],//
				//pw1:0
				//pw2:1,2
				//pw3:3 4 5 6 7 8 9 10
				//pw4: 11-42
				//pw5: 43-106
				//fc:  107-138
				//uint128 conv_weight_3x3_all[16][3][3],
				uint256 conv_weight_3x3_all[12][9],
				//dw1: 0
				//dw2: 1
				//dw3: 2 3
				//dw4: 4 5 6 7
				//dw5: 8 9 10 11 12 13 14 15 --> 8 9 10 11

				uint256 bias_all[31],
				//0       	:dw1,
				//1      	:pw1,
				//2			:dw2
				//3 4		:pw2
				//5 6		:dw3
				//7 8 9 10	:pw3
				//11 12 13 14:dw4
				//15-22		:pw4
				//23-30     :dw5
				//31-38		:pw5
				//39 40 41 42: fc
				uint256 DDR_dw1_pool_out_PL_burst[16/16*30*30],
				uint256 DDR_dw2_pool_out_PL_burst[32/16*16*16],
				uint256 DDR_buf_burst[20*9*9],
				//int *debug_pin,
				int cla[1]
)

{


#pragma HLS INTERFACE m_axi depth=3*58*58*8 		port=image_oringinal		    offset=slave	bundle=INPUT1
#pragma HLS INTERFACE m_axi depth=59*16*256   		port=conv_weight_1x1_all		offset=slave	bundle=INPUT2
#pragma HLS INTERFACE m_axi depth=12*9*256    		port=conv_weight_3x3_all		offset=slave	bundle=INPUT2
#pragma HLS INTERFACE m_axi depth=31*256			port=bias_all					offset=slave	bundle=INPUT2

#pragma HLS INTERFACE m_axi depth=30*30*256			port=DDR_dw1_pool_out_PL_burst	offset=slave	bundle=INPUT
#pragma HLS INTERFACE m_axi depth=2*16*16*256		port=DDR_dw2_pool_out_PL_burst	offset=slave	bundle=INPUT
#pragma HLS INTERFACE m_axi depth=20*9*9*256		port=DDR_buf_burst				offset=slave	bundle=INPUT




#pragma HLS INTERFACE m_axi depth=2*32				port=cla				offset=slave	bundle=OUTPUT


#pragma HLS INTERFACE s_axilite register	port=return


#pragma HLS ALLOCATION instances=Conv2D				 		limit=1 function
#pragma HLS ALLOCATION instances=Relu_Max_Pooling	    	limit=1 function
#pragma HLS ALLOCATION instances=global_pooling	    		limit=1 function
#pragma HLS ALLOCATION instances=load_image_normal		    limit=1 function



		int CI_N, CO_N;
		int weight_3x3_index, weight_1x1_index, bias_3x3_index, bias_1x1_index;

		/////////////////////////////// DW1+PW1 + POOL ////////////////////////////
        //INPOUT 3*58*58->16*28*28     to buf 4 5 6 7
		weight_3x3_index = 0;
		bias_3x3_index = 0;
		weight_1x1_index = 0;
		bias_1x1_index = 1;

		CI_N = 16 / 16;
		CO_N = 16 / 16;

		load_weight_3x3_from_axi(weight_buf[0], conv_weight_3x3_all[weight_3x3_index]);
		load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index]);
		load_bias_from_axi(bias_buf[1], bias_all[bias_1x1_index]); // maybe not 32
		load_weight_1x1_from_axi(weight_buf[1], conv_weight_1x1_all[weight_1x1_index]);
		#ifdef CSIM_DEBUG
		dw_weight_HLS_output(weight_buf[0],0);
		dw1_bias_weight_HLS_output(bias_buf[0]);
		pw1_bias_weight_HLS_output(bias_buf[1]);
		pw1_weight_HLS_output(weight_buf[1], 0);

		#endif

		for(int row = 0; row < 4; row++) {

			load_image_normal(FM_buf1, image_oringinal, 0, row);

			#ifdef CSIM_DEBUG
			image_HLS_output(FM_buf1,0,0, row);
			#endif

			for(int col = 0; col < 4; col++) {

				if( col % 2 == 0 ) {
					Conv2D(FM_buf10, FM_buf11, FM_buf1, FM_buf3, weight_buf[0], bias_buf[0], 0, 1);
					load_image_normal(FM_buf2, image_oringinal, col+1, row);
#ifdef CSIM_DEBUG
image_HLS_output(FM_buf2,0,col+1, row);
#endif
				}
				else {
					Conv2D(FM_buf10,FM_buf11, FM_buf2, FM_buf3, weight_buf[0], bias_buf[0], 0, 1);
					load_image_normal(FM_buf1, image_oringinal, col+1, row); // col=4
#ifdef CSIM_DEBUG
image_HLS_output(FM_buf1,0,col+1, row);
#endif

				}
				#ifdef CSIM_DEBUG
				fill_output_16(1, FM_buf3, 0, col, row);
				#endif

				for(int co = 0; co < CO_N; co++) {
					set_bias(FM_buf_acc, bias_buf[1 + co]);
					Conv2D(FM_buf10,FM_buf11, FM_buf3, FM_buf_acc, weight_buf[1], bias_buf[0], 1, 1);
					Relu_Max_Pooling(FM_buf_acc, DDR_dw1_pool_out_PL_burst, DDR_dw2_pool_out_PL_burst, DDR_buf_burst, 0, co, col, row, 1);
					#ifdef CSIM_DEBUG
					fill_output_16(2, FM_buf_acc, 0, col, row);
					#endif
				}
			}
		}	
		printf("DW1 Done\n");	

///////////////////////////// 2nd layer ////////////////////////////////////////
		//DW   input channel=16  size=28*28    out channel=16
        //pw    channel=32
		weight_3x3_index += CI_N; // 1
		bias_3x3_index  += (CI_N + CO_N); //2
		weight_1x1_index += (CO_N * CI_N); //1 2
		bias_1x1_index += (CO_N + CO_N ); //3 4

		load_weight_3x3_from_axi(weight_buf[0], conv_weight_3x3_all[weight_3x3_index]);

		//load_weight_3D_from_axi(weight_buf_3x3[0], conv_weight_3x3_all[weight_3x3_index]);
		load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index]);
				
		CI_N = 16 / 16;
		CO_N = 32 / 16;
		
		for(int row = 0; row < 2; row++) {			
			for(int col = 0; col < 2; col++) {

				load_dw1_pool_from_DDR(DDR_dw1_pool_out_PL_burst, FM_buf1, 0, col, row);
//				set_bias(FM_buf3, bias_buf[0]);
//				CONV_3x3_group(FM_buf1, FM_buf3, weight_buf_3x3[0]);
//				Relu(FM_buf3);

				Conv2D(FM_buf10,FM_buf11, FM_buf1, FM_buf3, weight_buf[0], bias_buf[0], 0, 1);
//				#ifdef CSIM_DEBUG  //print feature map
//				std::cout<<"row:"<<row<<"col:"<<col<<std::endl;
//				for(int i=0;i<16;i++){
//					for(int j=0;j<16;j++){
//						for(int k=0;k<16;k++){
//							std::cout<<" FM_buf3 i:"<<dec<<(i)<<" j:"<<dec<<j<<" k:"<<dec<<k<<" :"<<FM_buf3[i][j][k]<<"  "<<std::endl;
//
//						}
//						std::cout<<"\n"<<std::endl;
//					}
//					std::cout<<"\n"<<std::endl;
//				}
//				std::cout<<"\n"<<std::endl;
//				#endif

				#ifdef CSIM_DEBUG
				fill_output_16(4, FM_buf3, 0, col, row);
				fill_output_pool(3, FM_buf1,0,col,row);
				#endif

				for(int co = 0; co < CO_N; co++) {
					load_bias_from_axi(bias_buf[1], bias_all[bias_1x1_index + co]); //first 16 out
					set_bias(FM_buf_acc, bias_buf[1]);
					load_weight_1x1_from_axi(weight_buf[1], conv_weight_1x1_all[weight_1x1_index + co * CI_N]);


					//load_weight_1x1_from_axi(weight_buf_1x1[1], conv_weight_1x1_all[weight_1x1_index + co * CI_N]);//first 16 out

//					CONV_1x1(FM_buf3, FM_buf_acc, weight_buf_1x1[1]);
//					Relu(FM_buf_acc);

//					max_pooling(FM_buf_acc, FM_buf_pool);
//					copy_to_DDR_pool6( DDR_pool6_out_PL, FM_buf_pool, co, col, row);

					Conv2D(FM_buf10,FM_buf11, FM_buf3, FM_buf_acc, weight_buf[1], bias_buf[0], 1, 1);
					
					Relu_Max_Pooling(FM_buf_acc, DDR_dw1_pool_out_PL_burst, DDR_dw2_pool_out_PL_burst, DDR_buf_burst, 0, co, col, row, 2);
#ifdef CSIM_DEBUG
fill_output_16(5, FM_buf_acc, co, col, row);
#endif

				}
			}
		}
		printf("DW2 Done\n");		



///////////////////// 3rd layer/////////////////
//input buf , size 14*14 channel =32, out put 64 channel     size = 7*7
		
		weight_3x3_index += CI_N; //2,3
		bias_3x3_index  += (CI_N + CO_N); //5,6
		weight_1x1_index += (CO_N * CI_N); //3-10
		bias_1x1_index += (CO_N + CO_N); //7-10

		load_weight_3x3_from_axi(weight_buf[0], conv_weight_3x3_all[weight_3x3_index + 0]);
		load_weight_3x3_from_axi(weight_buf[1], conv_weight_3x3_all[weight_3x3_index + 1]);
		load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + 0]);
		load_bias_from_axi(bias_buf[1], bias_all[bias_3x3_index + 1]);

		CI_N = 32 / 16;
		CO_N = 64 / 16;

		load_dw2_pool_from_DDR(DDR_dw2_pool_out_PL_burst, FM_buf1, 0, 0, 0); // col row :0, 0
		Conv2D(FM_buf10,FM_buf11, FM_buf1, FM_buf2, weight_buf[0], bias_buf[0], 0, 1);
#ifdef CSIM_DEBUG
fill_output_16(7, FM_buf2, 0, 0, 0);
fill_output_pool(6, FM_buf1,0,0,0);
#endif

		//load_pool6_from_axi(FM_buf4, DDR_pool6_out_PL, 1, 0, 0);

		load_dw2_pool_from_DDR(DDR_dw2_pool_out_PL_burst, FM_buf4, 1, 0, 0);
		Conv2D(FM_buf10,FM_buf11, FM_buf4, FM_buf3, weight_buf[1], bias_buf[1], 0, 1);
#ifdef CSIM_DEBUG
fill_output_16(7, FM_buf3, 1, 0, 0);
fill_output_pool(6, FM_buf4,1,0,0);
#endif

		for(int co = 0; co < CO_N; co++) {
			load_bias_from_axi(bias_buf[3], bias_all[bias_1x1_index + co]); //7,8,9,10
			set_bias(FM_buf_acc, bias_buf[3]);

			load_weight_1x1_from_axi(weight_buf[2], conv_weight_1x1_all[weight_1x1_index + 0 + co * CI_N]);//3,5
			Conv2D(FM_buf10,FM_buf11, FM_buf2, FM_buf_acc, weight_buf[2], bias_buf[0], 1, 0);


			load_weight_1x1_from_axi(weight_buf[3], conv_weight_1x1_all[weight_1x1_index + 1 + co * CI_N]);//4,6
			Conv2D(FM_buf10,FM_buf11, FM_buf3, FM_buf_acc, weight_buf[3], bias_buf[0], 1, 1);
#ifdef CSIM_DEBUG
fill_output_16(8, FM_buf_acc, co, 0, 0);
#endif
			Relu_Max_Pooling(FM_buf_acc, DDR_dw1_pool_out_PL_burst, DDR_dw2_pool_out_PL_burst, DDR_buf_burst, co, 0, 0, 0, 3);

		}
		printf("DW3 Done\n");		



/////////		/////////////////changed 4th layer///////////////////////////////////////////////
		//input DDR_buf_burst:0,1,2,3
		
		// weight_3x3_index = 4; //4,5,6,7
		// bias_3x3_index = 11; // 11 12 13 14
		// weight_1x1_index = 11; //11-26
		// bias_1x1_index = 15; //15-18

		weight_3x3_index += CI_N; //2+2=4; 4-7
		bias_3x3_index  += (CI_N + CO_N); //5+2+4=11
		weight_1x1_index += (CO_N * CI_N); //3+2*4=11
		bias_1x1_index += (CO_N + CO_N); //7+8=15

		CI_N = 64 / 16;
		CO_N = 64 / 16;

		load_buf_from_DDR(FM_buf10, DDR_buf_burst, 0);
#ifdef CSIM_DEBUG
fill_output_pool9(9, FM_buf10, 0, 0, 0);
#endif
		for(int c = 0; c < CI_N; c++) {
			load_weight_3x3_from_axi(weight_buf[0], conv_weight_3x3_all[weight_3x3_index + c]);
			load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + c]);

			if( c % 2 == 0 ) {
				load_buf_from_DDR(FM_buf11, DDR_buf_burst, c+1);

				Conv2D(FM_buf10, FM_buf13, FM_buf1, FM_buf3, weight_buf[0], bias_buf[0], 2, 1);
#ifdef CSIM_DEBUG
fill_output_pool9(9, FM_buf11, c+1, 0, 0);
//fill_output_9(10,FM_buf13,c,0,0);
#endif
			}
			else {
				load_buf_from_DDR(FM_buf10, DDR_buf_burst, c+1);

				Conv2D(FM_buf11, FM_buf13, FM_buf1, FM_buf3, weight_buf[0], bias_buf[0], 2, 1);
#ifdef CSIM_DEBUG
fill_output_pool9(9, FM_buf10, c+1, 0, 0);
//fill_output_9(10,FM_buf13,c,0,0);
#endif
			}

			relu_copy_buf_to_DDR(DDR_buf_burst, 4 + c, FM_buf13); //DDR_buf_burst:4,5,6,7
		}

		for(int co = 0; co < CO_N; co++) {

			load_bias_from_axi(bias_buf[1], bias_all[bias_1x1_index + co]);
			set_bias7(FM_buf13, bias_buf[1]);

			load_buf_from_DDR(FM_buf10, DDR_buf_burst, 4 + 0);
#ifdef CSIM_DEBUG
fill_output_9(10,FM_buf10,0,0,0);
#endif
			for(int ci = 0; ci < CI_N; ci++) {
				load_weight_1x1_from_axi(weight_buf[1], conv_weight_1x1_all[weight_1x1_index + ci + co * CI_N]);
				if( ci % 2 == 0) {
					load_buf_from_DDR(FM_buf11, DDR_buf_burst, 4 + ci+1);
					Conv2D(FM_buf10,FM_buf13, FM_buf3, FM_buf_acc, weight_buf[1], bias_buf[0], 3, 0);
#ifdef CSIM_DEBUG
fill_output_9(10,FM_buf11,ci+1,0,0);
#endif
				}
				else {
					load_buf_from_DDR(FM_buf10, DDR_buf_burst, 4 + ci+1);
					Conv2D(FM_buf11,FM_buf13, FM_buf3, FM_buf_acc, weight_buf[1], bias_buf[0], 3, 0);
#ifdef CSIM_DEBUG
fill_output_9(10,FM_buf10,ci+1,0,0);
#endif
				}
				
			}
			Relu(FM_buf13);
			relu_copy_buf_to_DDR(DDR_buf_burst, 8 + co, FM_buf13); //DDR_buf_burst:8,9.10,11	
//#ifdef CSIM_DEBUG
//fill_output_9(11,FM_buf13,co,0,0);
//#endif
		}
		printf("DW4 Done\n");	

/////////		/////////////////changed 5th layer///////////////////////////////////////////////
		//input 10 11 12 13 out 18 19 20 21
		
		// weight_3x3_index = 8; //dw5: 8 9 10 11 
		// bias_3x3_index = 19; //19-22    
		// weight_1x1_index = 27; //pw5: 27-42
		// bias_1x1_index = 23; //23-26	

		weight_3x3_index += CI_N;  //4+4=8
		bias_3x3_index  += (CI_N + CO_N); //11+8=19
		weight_1x1_index += (CO_N * CI_N); //11+16=27
		bias_1x1_index += (CO_N + CO_N); //15+8=23 //this layer CO_N + last layer CO_N

		CI_N = 64 / 16;
		CO_N = 64 / 16;

//////////////////////////////////////////////////////////////////////////////////////
		load_buf_from_DDR(FM_buf10, DDR_buf_burst, 8);
#ifdef CSIM_DEBUG
fill_output_9(11,FM_buf10,0,0,0);
#endif
		for(int c = 0; c < CI_N; c++) {
			load_weight_3x3_from_axi(weight_buf[0], conv_weight_3x3_all[weight_3x3_index + c]);
			load_bias_from_axi(bias_buf[0], bias_all[bias_3x3_index + c]);

			if( c % 2 == 0 ) {
				load_buf_from_DDR(FM_buf11, DDR_buf_burst, c+9);

				Conv2D(FM_buf10, FM_buf13, FM_buf1, FM_buf3, weight_buf[0], bias_buf[0], 2, 1);
#ifdef CSIM_DEBUG
fill_output_9(11,FM_buf11,c+1,0,0);

#endif
			}
			else {
				load_buf_from_DDR(FM_buf10, DDR_buf_burst, c+9);

				Conv2D(FM_buf11, FM_buf13, FM_buf1, FM_buf3, weight_buf[0], bias_buf[0], 2, 1);
#ifdef CSIM_DEBUG
fill_output_9(11, FM_buf10, c+1, 0, 0);
#endif
			}

			relu_copy_buf_to_DDR(DDR_buf_burst, 12 + c, FM_buf13); //DDR_buf_burst:4,5,6,7
		}

		for(int co = 0; co < CO_N; co++) {
            clear_buffer(FM_buf13);

			load_bias_from_axi(bias_buf[1], bias_all[bias_1x1_index + co]);
			set_bias7(FM_buf13, bias_buf[1]);


			load_buf_from_DDR(FM_buf10, DDR_buf_burst, 12);
#ifdef CSIM_DEBUG
fill_output_9(12,FM_buf10,0,0,0);
conv13_bias_HLS_output(bias_buf[1],co);
#endif
			for(int ci = 0; ci < CI_N; ci++) {
				load_weight_1x1_from_axi(weight_buf[1], conv_weight_1x1_all[weight_1x1_index + ci + co * CI_N]);
				if( ci % 2 == 0) {
					load_buf_from_DDR(FM_buf11, DDR_buf_burst, ci+13);
					Conv2D(FM_buf10,FM_buf13, FM_buf3, FM_buf1, weight_buf[1], bias_buf[0], 3, 0);
#ifdef CSIM_DEBUG
fill_weight_output(13, weight_buf[1], co, ci);
fill_output_9(12,FM_buf11,ci+1,0,0);
#endif
				}
				else {
					load_buf_from_DDR(FM_buf10, DDR_buf_burst, ci+13);
					Conv2D(FM_buf11,FM_buf13, FM_buf3, FM_buf1, weight_buf[1], bias_buf[0], 3, 0);
#ifdef CSIM_DEBUG
fill_weight_output(13, weight_buf[1], co, ci);
fill_output_9(12,FM_buf10,ci+1,0,0);
#endif
				}

			}
			Relu(FM_buf13);
#ifdef CSIM_DEBUG
fill_output_9(13, FM_buf13, co, 0, 0);
conv13_weight_HLS_output();
#endif

//#ifdef CSIM_DEBUG  //print feature map
//if (co==0){
//	for(int i=0;i<16;i++){
//		for(int j=0;j<7;j++){
//			for(int k=0;k<7;k++){
//				std::cout<<" FM_buf13 i:"<<dec<<(i)<<" j:"<<dec<<j<<" k:"<<dec<<k<<" :"<<FM_buf13[i][j][k]<<"  "<<std::endl;
//
//			}
//			std::cout<<"\n"<<std::endl;
//		}
//		std::cout<<"\n"<<std::endl;
//	}
//	std::cout<<"\n"<<std::endl;
//}
//#endif
			global_pooling(FM_buf13, FC_IN[co]);


		}
		printf("DW5 Done\n");	




/////////fc
		weight_1x1_index += (CO_N * CI_N); //27+16=43, 43-58
		bias_1x1_index += (CO_N + CO_N); //23+4=27,do not have 3*3 bias

		/***** for(int i = 0; c < 4; c++) {

			load_bias_from_axi(bias_buf[i], bias_all[bias_1x1_index + i]); //first 16 out
			load_weight_1x1_from_axi(weight_buf[i], conv_weight_1x1_all[weight_1x1_index + i]);
			FC(FC_IN[i],weight_buf[i],bias_buf[i]);
		}
 *******/

		load_bias_from_axi(bias_buf[0], bias_all[bias_1x1_index + 0]); //first 16 out
		load_bias_from_axi(bias_buf[1], bias_all[bias_1x1_index + 1]);
		load_bias_from_axi(bias_buf[2], bias_all[bias_1x1_index + 2]);
		load_bias_from_axi(bias_buf[3], bias_all[bias_1x1_index + 3]);

		gap_fc(bias_buf[0],fc_buf[0]);
		gap_fc(bias_buf[1],fc_buf[1]);
		gap_fc(bias_buf[2],fc_buf[2]);
		gap_fc(bias_buf[3],fc_buf[3]);

//#ifdef CSIM_DEBUG
//		for(int i=0;i<16;i++){
//			std::cout<<"\n bias_buf0 HLS:   "<<dec<<(i)<<"    :"<<dec<<bias_buf[0][i]<<std::endl;
//		}
//		for(int i=0;i<16;i++){
//			std::cout<<"\n  bias_buf1 HLS:   "<<dec<<(i+16)<<"    :"<<dec<<bias_buf[1][i]<<std::endl;
//		}
//		for(int i=0;i<16;i++){
//			std::cout<<"\n  bias_buf2 HLS:   "<<dec<<(i+32)<<"    :"<<dec<<bias_buf[2][i]<<std::endl;
//		}
//
//		for(int i=0;i<16;i++){
//			std::cout<<"\n  bias_buf3 HLS:   "<<dec<<(i+48)<<"    :"<<dec<<bias_buf[3][i]<<std::endl;
//		}
//
//		for(int i=0;i<16;i++){
//			std::cout<<"\n fc_buf0_inital HLS:   "<<dec<<(i)<<"    :"<<dec<<fc_buf[0][i]<<std::endl;
//		}
//		for(int i=0;i<16;i++){
//			std::cout<<"\n  fc_buf1_inital HLS:   "<<dec<<(i+16)<<"    :"<<dec<<fc_buf[1][i]<<std::endl;
//		}
//		for(int i=0;i<16;i++){
//			std::cout<<"\n  fc_buf2_inital HLS:   "<<dec<<(i+32)<<"    :"<<dec<<fc_buf[2][i]<<std::endl;
//		}
//
//		for(int i=0;i<16;i++){
//			std::cout<<"\n  fc_out3_inital HLS:   "<<dec<<(i+48)<<"    :"<<dec<<fc_buf[3][i]<<std::endl;
//		}
//
//
//
//#endif
//1st
		load_weight_1x1_from_axi(weight_buf[0], conv_weight_1x1_all[weight_1x1_index + 0]);
		load_weight_1x1_from_axi(weight_buf[1], conv_weight_1x1_all[weight_1x1_index + 1]);
		load_weight_1x1_from_axi(weight_buf[2], conv_weight_1x1_all[weight_1x1_index + 2]);
		load_weight_1x1_from_axi(weight_buf[3], conv_weight_1x1_all[weight_1x1_index + 3]);

		FC(FC_IN[0],weight_buf[0],fc_buf[0]);
		FC(FC_IN[1],weight_buf[1],fc_buf[0]);
		FC(FC_IN[2],weight_buf[2],fc_buf[0]);
		FC(FC_IN[3],weight_buf[3],fc_buf[0]);

//2nd
		load_weight_1x1_from_axi(weight_buf[0], conv_weight_1x1_all[weight_1x1_index + 4]);
		load_weight_1x1_from_axi(weight_buf[1], conv_weight_1x1_all[weight_1x1_index + 5]);
		load_weight_1x1_from_axi(weight_buf[2], conv_weight_1x1_all[weight_1x1_index + 6]);
		load_weight_1x1_from_axi(weight_buf[3], conv_weight_1x1_all[weight_1x1_index + 7]);

		FC(FC_IN[0],weight_buf[0],fc_buf[1]);
		FC(FC_IN[1],weight_buf[1],fc_buf[1]);
		FC(FC_IN[2],weight_buf[2],fc_buf[1]);
		FC(FC_IN[3],weight_buf[3],fc_buf[1]);

//3rd
		load_weight_1x1_from_axi(weight_buf[0], conv_weight_1x1_all[weight_1x1_index + 8]);
		load_weight_1x1_from_axi(weight_buf[1], conv_weight_1x1_all[weight_1x1_index + 9]);
		load_weight_1x1_from_axi(weight_buf[2], conv_weight_1x1_all[weight_1x1_index + 10]);
		load_weight_1x1_from_axi(weight_buf[3], conv_weight_1x1_all[weight_1x1_index + 11]);

		FC(FC_IN[0],weight_buf[0],fc_buf[2]);
		FC(FC_IN[1],weight_buf[1],fc_buf[2]);
		FC(FC_IN[2],weight_buf[2],fc_buf[2]);
		FC(FC_IN[3],weight_buf[3],fc_buf[2]);

//4th
		load_weight_1x1_from_axi(weight_buf[0], conv_weight_1x1_all[weight_1x1_index + 12]);
		load_weight_1x1_from_axi(weight_buf[1], conv_weight_1x1_all[weight_1x1_index + 13]);
		load_weight_1x1_from_axi(weight_buf[2], conv_weight_1x1_all[weight_1x1_index + 14]);
		load_weight_1x1_from_axi(weight_buf[3], conv_weight_1x1_all[weight_1x1_index + 15]);

		FC(FC_IN[0],weight_buf[0],fc_buf[3]);
		FC(FC_IN[1],weight_buf[1],fc_buf[3]);
		FC(FC_IN[2],weight_buf[2],fc_buf[3]);
		FC(FC_IN[3],weight_buf[3],fc_buf[3]);

#ifdef CSIM_DEBUG
		fill_output_fc(fc_buf[0], 0);
		fill_output_fc(fc_buf[1], 1);
		fill_output_fc(fc_buf[2], 2);
		fill_output_fc(fc_buf[3], 3);
		fill_output_gap(FC_IN[0], 0);
		fill_output_gap(FC_IN[1], 1);
		fill_output_gap(FC_IN[2], 2);
		fill_output_gap(FC_IN[3], 3);
#endif



//#ifdef CSIM_DEBUG
//		//int sum=0;
//
////		for(int j=0;j<4;j++){
////			for(int i=0;i<16;i++){
////				std::cout<<"\n gap output:   "<<dec<<(j*16+i)<<"    :"<<FC_IN[j][i]<<std::endl;
////			}
////		}
//
//		for(int i=0;i<16;i++){
//			std::cout<<"\n fc_buf1_last fc_out HLS:   "<<dec<<(i)<<"    :"<<fc_buf[0][i]<<std::endl;
//		}
//		for(int i=0;i<16;i++){
//			std::cout<<"\n fc_buf2_last fc_out HLS:   "<<dec<<(i+16)<<"    :"<<fc_buf[1][i]<<std::endl;
//		}
//		for(int i=0;i<16;i++){
//			std::cout<<"\n fc_buf3_last fc_out HLS:   "<<dec<<(i+32)<<"    :"<<fc_buf[2][i]<<std::endl;
//		}
//
//		for(int i=0;i<16;i++){
//			std::cout<<"\n fc_buf4_last fc_out HLS:   "<<dec<<(i+48)<<"    :"<<fc_buf[3][i]<<std::endl;
//		}
//
//
//
//#endif
#ifdef CSIM_DEBUG
		PL_golden_compare_layer_1();
		PL_golden_compare_layer_2();
		PL_golden_compare_layer_3();
		PL_golden_compare_layer_4();
		PL_golden_compare_layer_5();
		PL_golden_compare_layer_6();
		PL_golden_compare_layer_7();
		PL_golden_compare_layer_8();
		PL_golden_compare_layer_9();
		PL_golden_compare_layer_10();
		PL_golden_compare_layer_11();
		PL_golden_compare_layer_12();

		PL_golden_compare_layer_13();
		PL_golden_compare_layer_gap();
		PL_golden_compare_layer_fc();


#endif

		sort(fc_buf[0],fc_buf[1],fc_buf[2],fc_buf[3],cla);
		printf("sort Done\n");


}



