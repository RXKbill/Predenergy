#include "lightsensor.h"
#include "main.h"




void GET_RAIN_SENSOR(void){
	uint8_t 	SEND_Buf[40];		//发送给传感器的BUF	
	memset(SEND_Buf,0,40);
	SEND_Buf[0]=0X01;
	SEND_Buf[1]=0X03;
	SEND_Buf[2]=0X00;
	SEND_Buf[3]=0X00;
	SEND_Buf[4]=0X00;
	SEND_Buf[5]=0X01;
	SEND_Buf[6]=0X84;
	SEND_Buf[7]=0X0A;
	HAL_UART_Transmit(&huart3, SEND_Buf,8, 1000);	
	RAIN_SENSOR_REC_CHECK();
	
}




uint8_t RAIN_SENSOR_REC_CHECK(void){
	uint16_t count;
	uint16_t i;	
	count=0;
	while((Res3_Sign==0)&&(count<50)){//等待响应
		count++;
		HAL_Delay(10);
	}
	if(count==50){// 超时
		return 0;
	}else{ // 接收到响应
		do{// 接收到最后一个字节 再延时10ms
				Res3++;
				HAL_Delay(1);
		}while(Res3<10);
		
//		HAL_UART_Transmit(&huart1, Res3_Buf, Res3_Count, 1000);		//转发到串口1			

		for(i=0;i<Res3_Count;i++){
			if(Res3_Buf[i]==0x01&&Res3_Buf[i+1]==0X03&Res3_Buf[i+2]==0x02){
				
			RAIN_INTENSITY_H=Res3_Buf[i+3];
			RAIN_INTENSITY_L=Res3_Buf[i+4];
				
			memset(Res3_Buf,0,256);
			Res3_Sign=0;	
			Res3_Count=0;				
			return 1;
			}	
		}
	}
		memset(Res3_Buf,0,256);
		Res3_Sign=0;	
		Res3_Count=0;		
		return 0;
}





