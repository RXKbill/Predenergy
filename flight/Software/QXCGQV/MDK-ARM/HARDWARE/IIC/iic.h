#ifndef __MPUIIC_H
#define __MPUIIC_H
#include "main.h"
   
 
#define IIC_SCL_Pin GPIO_PIN_6
#define IIC_SCL_GPIO_Port GPIOB
#define IIC_SDA_Pin GPIO_PIN_7
#define IIC_SDA_GPIO_Port GPIOB 
	 
//IO方向设置
//#define SDA_IN()  {GPIOB->CRL&=0X0FFFFFFF;GPIOB->CRL|=(U32)8<<28;}	//PB7输入模式
//#define SDA_OUT() {GPIOB->CRL&=0X0FFFFFFF;GPIOB->CRL|=(U32)3<<28;} 	//PB7输出模式

void IIC_SDA_SET(int mode);
void IIC_SCL_SET(int mode);
//void SDA_OUT(void);
//void SDA_IN(void);
uint8_t IIC_SDA_READ(void);
void NOISE_FILTER(void);
void SDA_IN(void);	
void SDA_OUT(void);	
void MPU_IIC_Delay(void);				//MPU IIC延时函数
void MPU_IIC_Init(void);        //初始化IIC的IO口				 
void MPU_IIC_Start(void);				//发送IIC开始信号
void MPU_IIC_Stop(void);	  	  //发送IIC停止信号
void MPU_IIC_Send_Byte(uint8_t txd);	//IIC发送一个字节
uint8_t MPU_IIC_Read_Byte(unsigned char ack);//IIC读取一个字节
uint8_t MPU_IIC_Wait_Ack(void); 				//IIC等待ACK信号
void MPU_IIC_Ack(void);					//IIC发送ACK信号
void MPU_IIC_NAck(void);				//IIC不发送ACK信号
void IMPU_IC_Write_One_Byte(uint8_t daddr,uint8_t addr,uint8_t data);
uint8_t MPU_IIC_Read_One_Byte(uint8_t daddr,uint8_t addr);	  
#endif
















