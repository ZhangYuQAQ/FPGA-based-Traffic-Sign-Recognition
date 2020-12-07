//#include "dscnet_8.h"
#include "dscnet_16.h"

#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"

using namespace std;

//mode = 0 3x3depthwise; mode = 1 1x1 conv2d
#define X1 1
#define X3 0
//inline FIX_FM relu_single( FIX_FM d ) {
//	if( d < 0 )
//		return 0;
//	else
//		return d;
//}
FIX_32_12 compute_engine_16(FIX_WT w0,  FIX_FM b0,
					  FIX_WT w1,  FIX_FM b1,
					  FIX_WT w2,  FIX_FM b2,
					  FIX_WT w3,  FIX_FM b3,
					  FIX_WT w4,  FIX_FM b4,
					  FIX_WT w5,  FIX_FM b5,
					  FIX_WT w6,  FIX_FM b6,
					  FIX_WT w7,  FIX_FM b7,
					  FIX_WT w8,  FIX_FM b8,
					  FIX_WT w9,  FIX_FM b9,
					  FIX_WT w10, FIX_FM b10,
					  FIX_WT w11, FIX_FM b11,
					  FIX_WT w12, FIX_FM b12,
					  FIX_WT w13, FIX_FM b13,
					  FIX_WT w14, FIX_FM b14,
					  FIX_WT w15, FIX_FM b15)
{
	FIX_32_10 mul0, mul1, mul2,  mul3,  mul4,  mul5,  mul6,  mul7;
	FIX_32_10 mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15;
	FIX_32_10 add0, add1, add2, add3,  add4,  add5,  add6;
	FIX_32_10 add7, add8, add9, add10, add11, add12, add13, add14;

	mul0  = w0  * b0;
	mul1  = w1  * b1;
	mul2  = w2  * b2;
	mul3  = w3  * b3;
	mul4  = w4  * b4;
	mul5  = w5  * b5;
	mul6  = w6  * b6;
	mul7  = w7  * b7;
	mul8  = w8  * b8;
	mul9  = w9  * b9;
	mul10 = w10 * b10;
	mul11 = w11 * b11;
	mul12 = w12 * b12;
	mul13 = w13 * b13;
	mul14 = w14 * b14;
	mul15 = w15 * b15;

	add0 = mul0  + mul1;
	add1 = mul2  + mul3;
	add2 = mul4  + mul5;
	add3 = mul6  + mul7;
	add4 = mul8  + mul9;
	add5 = mul10 + mul11;
	add6 = mul12 + mul13;
	add7 = mul14 + mul15;

	add8  = add0 + add1;
	add9  = add2 + add3;
	add10 = add4 + add5;
	add11 = add6 + add7;

	add12 = add8  + add9;
	add13 = add10 + add11;

	add14 = add12 + add13;

	return add14;

}

//FIX_32_12 compute_engine_9(FIX_WT w0,  FIX_FM b0,
//					  FIX_WT w1,  FIX_FM b1,
//					  FIX_WT w2,  FIX_FM b2,
//					  FIX_WT w3,  FIX_FM b3,
//					  FIX_WT w4,  FIX_FM b4,
//					  FIX_WT w5,  FIX_FM b5,
//					  FIX_WT w6,  FIX_FM b6,
//					  FIX_WT w7,  FIX_FM b7,
//					  FIX_WT w8,  FIX_FM b8
//					  )
//{
//	FIX_32_12 mul0, mul1, mul2,  mul3,  mul4,  mul5,  mul6,  mul7, mul8;
//	FIX_32_12 add0, add1, add2, add3,  add4,  add5,  add6, add7, add8;
//
//	mul0  = w0  * b0;
//	mul1  = w1  * b1;
//	mul2  = w2  * b2;
//	mul3  = w3  * b3;
//	mul4  = w4  * b4;
//	mul5  = w5  * b5;
//	mul6  = w6  * b6;
//	mul7  = w7  * b7;
//	mul8  = w8  * b8;
//
//	add0 = mul0  + mul1;
//	add1 = mul2  + mul3;
//	add2 = mul4  + mul5;
//	add3 = mul6  + mul7;
//
//	add4 = mul8;
//
//	add5  = add0 + add1;
//	add6  = add2 + add3;
//
//	add7 = add5 + add6;
//
//	add8 = add4 + add7;
//
//
//	return add8;
//
//}


void load_weights(FIX_WT weights[16][16], FIX_WT weight_buf[16][16])
{
	for(int ci = 0; ci < 16; ci++) {
//#pragma HLS pipeline
		for(int co = 0; co < 16; co++) {
//#pragma HLS unroll
#pragma HLS pipeline
			weight_buf[co][ci] = weights[co][ci];
		}
	}
}

void load_bias(FIX_WT bias[16], FIX_WT bias_buf[16])
{
	for(int co = 0; co < 16; co++) {
//#pragma HLS unroll
#pragma HLS pipeline
		bias_buf[co] = bias[co];
	}
}

/***********************************************
 * min:0.7w  max:1.4w
 * dsp:258 FF:22% LUT:62%
 * success
 *********************************************/
void Conv2D(
		FIX_FM in7[CHA][HEI_9][WID_9],
		FIX_FM out7[CHA][HEI_9][WID_9],
		FIX_FM in[CHA][HEI][WID],
		FIX_FM out[CHA][HEI][WID],
		FIX_WT weights[CHA][CHA],
		FIX_WT bias[CHA],
    	uint2 mode, uint1 relu) //mode = 0 3x3depthwise; mode = 1 1x1 conv2d
{

#pragma HLS array_partition variable=in dim=1 complete
#pragma HLS array_partition variable=out dim=1 complete

#pragma HLS array_partition variable=in7 dim=1 complete
#pragma HLS array_partition variable=out7 dim=1 complete

#pragma HLS array_partition variable=weights dim=1 complete
#pragma HLS array_partition variable=bias dim=1 complete

FIX_WT weight_buf[16][16];
#pragma HLS array_partition variable=weight_buf dim=1 complete
#pragma HLS array_partition variable=weight_buf dim=2 complete
FIX_WT bias_buf[16];
#pragma HLS array_partition variable=bias_buf dim=1 complete

//#pragma HLS ALLOCATION instances=compute_engine_9 limit=16 function
#pragma HLS ALLOCATION instances=compute_engine_16 limit=16 function

FIX_FM window[CHA][3][3];
#pragma HLS array_partition variable=window dim=0 complete

FIX_FM line_buffer[CHA][2][WID];
#pragma HLS array_partition variable=line_buffer dim=1 complete
FIX_FM zero=0;
FIX_WT zerowt=0;
//#pragma HLS DEPENDENCE variable=line_buffer inter false
FIX_32_12  tmp[16];
FIX_FM  tmp2;
#pragma HLS ARRAY_PARTITION variable=tmp dim=1 complete

	if(mode==X3){
		load_weights(weights, weight_buf);
		load_bias(bias, bias_buf);
		for(int h = 0; h < HEI; h++){
			for(int w = 0; w < WID; w++) {
#pragma HLS pipeline II=1
				for(int coo = 0; coo < CHA; coo++) {
#pragma HLS unroll
					for (int i=0;i<3;i++){
						window[coo][i][0] = window[coo][i][1];
						window[coo][i][1] = window[coo][i][2];
					}

					window[coo][0][2] = (line_buffer[coo][0][w]);
					window[coo][1][2] = (line_buffer[coo][0][w] = line_buffer[coo][1][w]);
					window[coo][2][2] = (line_buffer[coo][1][w] = in[coo][h][w]);


					if ((2<=h) && (2<=w)){

						out[coo][h-1][w-1] = bias_buf[coo] + (FIX_FM)compute_engine_16(
							weight_buf[coo][0],   window[coo][0][0], //84*(h+(ci/16)-1)+
							weight_buf[coo][1],   window[coo][0][1], //ex_bot1[coo][h][w-1]
							weight_buf[coo][2],   window[coo][0][2],
							weight_buf[coo][3],   window[coo][1][0],
							weight_buf[coo][4],   window[coo][1][1],
							weight_buf[coo][5],   window[coo][1][2],
							weight_buf[coo][6],   window[coo][2][0],
							weight_buf[coo][7],   window[coo][2][1],
							weight_buf[coo][8],   window[coo][2][2],
							zerowt,zero,
							zerowt,zero,
							zerowt,zero,
							zerowt,zero,
							zerowt,zero,
							zerowt,zero,
							zerowt,zero
							);
						out[coo][h-1][w-1] = (relu==1 && out[coo][h-1][w-1]<0)?zero:out[coo][h-1][w-1];
						//bottom[coo+ad][h-1][w-1] = (relu==1 && temp <0)?zero:temp;
						//FM 14 padding to 16, output:14,padding to 16
					}
				}
			}
		}
	}
	else if(mode==1){

		load_weights(weights, weight_buf);
		for(int h = 1; h <= (HEI-2); h++){
			for(int w = 1; w <= (WID-2); w++) {
#pragma HLS pipeline II=1
				for(int coo = 0; coo < CHA; coo++) {
#pragma HLS unroll
					tmp2 = out[coo][h][w]+compute_engine_16(
											weight_buf[coo][0],   in[0][h][w],
											weight_buf[coo][1],   in[1][h][w],
											weight_buf[coo][2],   in[2][h][w],
											weight_buf[coo][3],   in[3][h][w],
											weight_buf[coo][4],   in[4][h][w],
											weight_buf[coo][5],   in[5][h][w],
											weight_buf[coo][6],   in[6][h][w],
											weight_buf[coo][7],   in[7][h][w],
											weight_buf[coo][8],   in[8][h][w],
											weight_buf[coo][9],   in[9][h][w],
											weight_buf[coo][10],  in[10][h][w],
											weight_buf[coo][11],  in[11][h][w],
											weight_buf[coo][12],  in[12][h][w],
											weight_buf[coo][13],  in[13][h][w],
											weight_buf[coo][14],  in[14][h][w],
											weight_buf[coo][15],  in[15][h][w]);
					out[coo][h][w]=tmp2;
					out[coo][h][w] = (relu==1 && out[coo][h][w]<0)?zero:out[coo][h][w];
				}
			}
		}
	}
	else if(mode==2){
			load_weights(weights, weight_buf);
			load_bias(bias, bias_buf);
			for(int h = 0; h < HEI_9; h++){
				for(int w = 0; w < WID_9; w++) {
#pragma HLS pipeline II=1
					for(int coo = 0; coo < CHA; coo++) {
#pragma HLS unroll

						for (int i=0;i<3;i++){
							window[coo][i][0] = window[coo][i][1];
							window[coo][i][1] = window[coo][i][2];
						}

						window[coo][0][2] = (line_buffer[coo][0][w]);
						window[coo][1][2] = (line_buffer[coo][0][w] = line_buffer[coo][1][w]);
						window[coo][2][2] = (line_buffer[coo][1][w] = in7[coo][h][w]);


						if ((2<=h) && (2<=w)){

							out7[coo][h-1][w-1] = bias_buf[coo] + (FIX_FM)compute_engine_16(
								weight_buf[coo][0],   window[coo][0][0], //84*(h+(ci/16)-1)+
								weight_buf[coo][1],   window[coo][0][1], //ex_bot1[coo][h][w-1]
								weight_buf[coo][2],   window[coo][0][2],
								weight_buf[coo][3],   window[coo][1][0],
								weight_buf[coo][4],   window[coo][1][1],
								weight_buf[coo][5],   window[coo][1][2],
								weight_buf[coo][6],   window[coo][2][0],
								weight_buf[coo][7],   window[coo][2][1],
								weight_buf[coo][8],   window[coo][2][2],
								zerowt,zero,
								zerowt,zero,
								zerowt,zero,
								zerowt,zero,
								zerowt,zero,
								zerowt,zero,
								zerowt,zero
								);
							out7[coo][h-1][w-1] = (relu==1 && out7[coo][h-1][w-1]<0)?zero:out7[coo][h-1][w-1];
							//bottom[coo+ad][h-1][w-1] = (relu==1 && temp <0)?zero:temp;
						}
					}
				}
			}
		}
		else {

			load_weights(weights, weight_buf);
			for(int h = 1; h <= (HEI_9-2); h++){
				for(int w = 1; w <= (WID_9-2); w++) {
	#pragma HLS pipeline II=1
					for(int coo = 0; coo < CHA; coo++) {
	#pragma HLS unroll
							tmp2 = out7[coo][h][w]+compute_engine_16(
										weight_buf[coo][0],   in7[0][h][w],
										weight_buf[coo][1],   in7[1][h][w],
										weight_buf[coo][2],   in7[2][h][w],
										weight_buf[coo][3],   in7[3][h][w],
										weight_buf[coo][4],   in7[4][h][w],
										weight_buf[coo][5],   in7[5][h][w],
										weight_buf[coo][6],   in7[6][h][w],
										weight_buf[coo][7],   in7[7][h][w],
										weight_buf[coo][8],   in7[8][h][w],
										weight_buf[coo][9],   in7[9][h][w],
										weight_buf[coo][10],  in7[10][h][w],
										weight_buf[coo][11],  in7[11][h][w],
										weight_buf[coo][12],  in7[12][h][w],
										weight_buf[coo][13],  in7[13][h][w],
										weight_buf[coo][14],  in7[14][h][w],
										weight_buf[coo][15],  in7[15][h][w]);
							out7[coo][h][w]=tmp2;
							out7[coo][h][w] = (relu==1 && out7[coo][h][w]<0)?zero:out7[coo][h][w];
					}

				}
			}
		}
}

