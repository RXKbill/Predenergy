#include "uart.h"
#include "main.h"
#include "led.h"


void GET_RES2_BUF(void){
	  do{//接收到最后一个字节 再延时10ms
			Res2++;
			HAL_Delay(1);
		}while(Res2<4);

//		LED=!LED;		
		GET_UART2_NEWINFO();
//		HAL_UART_Transmit(&huart1, Res2_Buf, Res2_Count, 1000);		//转发到串口2	
		memset(Res2_Buf,0,256);
		Res2_Sign=0;	
	  Res2_Count=0;

}


void GET_RES3_BUF(void){
	  do{//接收到最后一个字节 再延时10ms
			Res3++;
			HAL_Delay(1);
		}while(Res3<4);
		
		

		
//		LED=!LED;
//		HAL_UART_Transmit(&huart1, Res3_Buf, Res3_Count, 1000);		//转发到串口2		
		memset(Res3_Buf,0,256);
		Res3_Sign=0;	
	  Res3_Count=0;
}



















