#include "main.h" 
#include "iic.h"



void MPU_IIC_Delay(void){  //MPU IIC 延时函数
	delay_us(2);
}


void SDA_OUT(void){
	GPIO_InitTypeDef GPIO_Initure;
	GPIO_Initure.Pin=GPIO_PIN_7;
	GPIO_Initure.Mode=GPIO_MODE_OUTPUT_PP;  //推挽输出
	GPIO_Initure.Pull=GPIO_PULLUP;          //上拉
	GPIO_Initure.Speed=GPIO_SPEED_FREQ_HIGH;//高速
	HAL_GPIO_Init(GPIOB,&GPIO_Initure);
}

void SDA_IN(void){
	GPIO_InitTypeDef GPIO_Initure;
	GPIO_Initure.Pin=GPIO_PIN_7;
	GPIO_Initure.Mode=GPIO_MODE_INPUT; 			//输入
	GPIO_Initure.Pull=GPIO_PULLUP;          //上拉
	GPIO_Initure.Speed=GPIO_SPEED_FREQ_HIGH;//高速
	HAL_GPIO_Init(GPIOB,&GPIO_Initure);
}	

void MPU_IIC_Init(void){//初始化IIC
					     
	GPIO_InitTypeDef GPIO_Initure;
	
	__HAL_RCC_GPIOB_CLK_ENABLE();           //开启GPIOA时钟
	 
    GPIO_Initure.Pin=GPIO_PIN_6|GPIO_PIN_7; //PB6 PB7
    GPIO_Initure.Mode=GPIO_MODE_OUTPUT_PP;  //推挽输出
    GPIO_Initure.Pull=GPIO_PULLUP;          //上拉
    GPIO_Initure.Speed=GPIO_SPEED_HIGH;    	//高速
	
    HAL_GPIO_WritePin(GPIOB,GPIO_PIN_6|GPIO_PIN_7,GPIO_PIN_SET);//PB6,PB7 输出高	
	
	HAL_GPIO_Init(GPIOB,&GPIO_Initure);
 
}

void MPU_IIC_Start(void){//产生IIC起始信号

	SDA_OUT();     //sda线输出
	IIC_SDA_SET(1);	  	  
	IIC_SCL_SET(1);
	MPU_IIC_Delay();
 	IIC_SDA_SET(0);//START:when CLK is high,DATA change form high to low 
	MPU_IIC_Delay();
	IIC_SCL_SET(0);//钳住I2C总线，准备发送或接收数据 
}	  

void MPU_IIC_Stop(void){//产生IIC停止信号
	SDA_OUT();//sda线输出
	IIC_SCL_SET(0);
	IIC_SDA_SET(0);//STOP:when CLK is high DATA change form low to high
 	MPU_IIC_Delay();
	IIC_SCL_SET(1);  
	IIC_SDA_SET(1);//发送I2C总线结束信号
	MPU_IIC_Delay();							   	
}

U8 MPU_IIC_Wait_Ack(void){//1，接收应答失败0，接收应答成功

	U8 ucErrTime=0;
	SDA_IN();      //SDA设置为输入  
	IIC_SDA_SET(1);MPU_IIC_Delay();	   
	IIC_SCL_SET(1);MPU_IIC_Delay();	 
	while(IIC_SDA_READ())
	{
		ucErrTime++;
		if(ucErrTime>250)
		{
			MPU_IIC_Stop();
			return 1;
		}
	}
	IIC_SCL_SET(0);//时钟输出0 	   
	return 0;  
} 

void MPU_IIC_Ack(void){//产生ACK应答

	IIC_SCL_SET(0);
	SDA_OUT();
	IIC_SDA_SET(0);
	MPU_IIC_Delay();
	IIC_SCL_SET(1);
	MPU_IIC_Delay();
	IIC_SCL_SET(0);
}
    
void MPU_IIC_NAck(void){//不产生ACK应答		

	IIC_SCL_SET(0);
	SDA_OUT();
	IIC_SDA_SET(1);
	MPU_IIC_Delay();
	IIC_SCL_SET(1);
	MPU_IIC_Delay();
	IIC_SCL_SET(0);
}					 				     


  
void MPU_IIC_Send_Byte(U8 txd){ //IIC发送一个字节//返回从机有无应答 //1，有应答          //0，无应答			            
    U8 t;   
	SDA_OUT(); 	    
    IIC_SCL_SET(0);//拉低时钟开始数据传输
    for(t=0;t<8;t++)
    {              
        IIC_SDA_SET((txd&0x80)>>7);
        txd<<=1; 	  
		IIC_SCL_SET(1);
		MPU_IIC_Delay(); 
		IIC_SCL_SET(0);	
		MPU_IIC_Delay();
    }	 
} 	    

U8 MPU_IIC_Read_Byte(unsigned char ack){//读1个字节，ack=1时，发送ACK，ack=0，发送nACK   
	unsigned char i,receive=0;
	SDA_IN();//SDA设置为输入
    for(i=0;i<8;i++ )
	{
        IIC_SCL_SET(0); 
        MPU_IIC_Delay();
		IIC_SCL_SET(1);
        receive<<=1;
        if(IIC_SDA_READ())receive++;   
		MPU_IIC_Delay(); 
    }					 
    if (!ack)
        MPU_IIC_NAck();//发送nACK
    else
        MPU_IIC_Ack(); //发送ACK   
    return receive;
}


void IIC_SDA_SET(int mode){
	if(mode){
		HAL_GPIO_WritePin(IIC_SDA_GPIO_Port, IIC_SDA_Pin, GPIO_PIN_SET); 
	}else HAL_GPIO_WritePin(IIC_SDA_GPIO_Port, IIC_SDA_Pin, GPIO_PIN_RESET);
}

uint8_t IIC_SDA_READ(void){
	uint8_t temp;
	temp=HAL_GPIO_ReadPin(IIC_SDA_GPIO_Port, IIC_SDA_Pin);
	return temp;
}
void IIC_SCL_SET(int mode){
	if(mode){
		HAL_GPIO_WritePin(IIC_SCL_GPIO_Port, IIC_SCL_Pin, GPIO_PIN_SET); 
	}else HAL_GPIO_WritePin(IIC_SCL_GPIO_Port, IIC_SCL_Pin, GPIO_PIN_RESET);
}




