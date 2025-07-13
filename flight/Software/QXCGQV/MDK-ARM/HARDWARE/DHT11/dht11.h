#ifndef __DHT11_H
#define __DHT11_H
#include "main.h"

extern	uint8_t temperature;
extern	uint8_t humidity;
	
//IO方向设置
#define DHT11_IO_IN(void)  {GPIOA->CRL&=0XFFFFFF0F;GPIOA->CRL|=8<<4;}
#define DHT11_IO_OUT(void) {GPIOA->CRL&=0XFFFFFF0F;GPIOA->CRL|=3<<4;}

//IO操作函数											   
#define	DHT11_DQ_OUT    PAout(1)//数据端口	PA1
#define	DHT11_DQ_IN     PAin(1) //数据端口	PA1

void GET_DHT11_Value(void);



uint8_t DHT11_Init(void);//初始化DHT11
uint8_t DHT11_Read_Data(uint8_t *temp,uint8_t *humi);//读取温湿度
uint8_t DHT11_Read_Byte(void);//读出一个字节
uint8_t DHT11_Read_Bit(void);//读出一个位
uint8_t DHT11_Check(void);//检测是否存在DHT11
void DHT11_Rst(void);//复位DHT11  
#endif
