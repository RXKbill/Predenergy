#ifndef __LINKAGE__
#define __LINKAGE__
#include "main.h"  



#define MODBUS_ID 0X90

#define NOISE_VALUE_H 										LINKAGE_BUF[0X10]			
#define NOISE_VALUE_L 										LINKAGE_BUF[0X11]

#define RAIN_INTENSITY_H									LINKAGE_BUF[0X18]
#define RAIN_INTENSITY_L									LINKAGE_BUF[0X19]

#define TEMP_H 														LINKAGE_BUF[0X1A]
#define TEMP_L														LINKAGE_BUF[0X1B]

#define HUMI_H														LINKAGE_BUF[0X1C]
#define HUMI_L														LINKAGE_BUF[0X1D]

#define PM1_H															LINKAGE_BUF[0X20]
#define PM1_L															LINKAGE_BUF[0X21]	
#define PM2_5_H														LINKAGE_BUF[0X22]	
#define PM2_5_L														LINKAGE_BUF[0X23]	
#define PM10_H														LINKAGE_BUF[0X24]	
#define PM10_L														LINKAGE_BUF[0X25]	
#define N_PM1_H														LINKAGE_BUF[0X26]	
#define N_PM1_L														LINKAGE_BUF[0X27]	
#define N_PM2_5_H													LINKAGE_BUF[0X28]	
#define N_PM2_5_L													LINKAGE_BUF[0X29]	
#define N_PM10_H													LINKAGE_BUF[0X2A]	
#define N_PM10_L													LINKAGE_BUF[0X2B]	
#define _0_3_NUM_H												LINKAGE_BUF[0X2C]	
#define _0_3_NUM_L												LINKAGE_BUF[0X2D]	
#define _0_5_NUM_H												LINKAGE_BUF[0X2E]	
#define _0_5_NUM_L												LINKAGE_BUF[0X2F]
#define _1_0_NUM_H												LINKAGE_BUF[0X30]	
#define _1_0_NUM_L												LINKAGE_BUF[0X31]	
#define _2_5_NUM_H												LINKAGE_BUF[0X32]	
#define _2_5_NUM_L												LINKAGE_BUF[0X33]	
#define _5_0_NUM_H												LINKAGE_BUF[0X34]			
#define _5_0_NUM_L												LINKAGE_BUF[0X35]			
#define _10_0_NUM_H												LINKAGE_BUF[0X36]		
#define _10_0_NUM_L												LINKAGE_BUF[0X37]		


extern uint8_t LINKAGE_BUF[200];

uint8_t HWDG_RELOAD(uint8_t count);
uint8_t BEEP_LOAD(uint8_t count);
void SLAVER_RST(void);	
	
	
unsigned int calccrc(unsigned char crcbuf,unsigned int crc); 
unsigned int chkcrc(unsigned char *buf,unsigned char len); 

void GET_RES1_BUF(void);
uint8_t GET_UART1_NEWINFO(void);



#endif


