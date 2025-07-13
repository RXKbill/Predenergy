#include "main.h" 
#include "linkage.h"


uint8_t LINKAGE_BUF[200];
//地址							功能																					格式														读写类型
//0X00 							NC																																							R/W	
//0X01							NC																																							R/W
//0X02							NC																																							R/W
//0X03							NC																																							R/W
//0X04							NC																																							R/W
//0X05							NC																																							R/W
//0X06							NC																																							R/W
//0X07							NC																																							R/W
//0X08							NC																																							R/W	
//0X09							NC																																							R/W
//0X0A							NC																																							R/W
//0X0B							NC																																							R/W
//0X0C							NC																																							R/W
//0X0D							NC																																							R/W									
//0X0E							NC																																							R/W
//0X0F							NC																																							R/W
//————————————————————————————————————————————————————————————————————
//0X10							噪声数值													dB					H																	R
//0X11																														L																	R
//0X12							NC																																							
//0X13							NC																																							
//0X14							NC																																							
//0X15							NC																																							
//0X16							NC																																							
//0X17				  		NC																																							
//0X18				  		雨量																					H																	R
//0X19				  																									L																	R
//0X1A				  		温度														0.01℃				H																	R
//0X1B				  																									L																	R
//0X1C				  		湿度																					H																	R
//0X1D				  																									L																	R
//0X1E				  		NC																																								
//0X1F				  		NC																																								
//————————————————————————————————————————————————————————————————————
//0X20						 	PM1.0 浓度（CF=1，标准颗粒物）		μ g/m3				H 																R
//0X21																														L 														    R
//0X22							PM2.5 浓度（CF=1，标准颗粒物）		μ g/m3				H 														    R
//0X23																														L 														    R
//0X24							PM10 浓度（CF=1，标准颗粒物）			μ g/m3				H 														    R
//0X25																														L 														    R
//0X26						 	PM1.0 浓度（大气环境下）					μ g/m3				H 														    R
//0X27																														L 														    R
//0X28							PM2.5 浓度（大气环境下）					μ g/m3				H 														    R
//0X29																														L 														    R
//0X2A						 	PM10 浓度 （大气环境下）					μ g/m3				H 														    R
//0X2B																														L 														    R
//0X2C						 	0.1 L空气中直径在0.3um以上颗粒物个数						H 														    R
//0X2D																														L 														    R
//0X2E							0.1 L空气中直径在0.5um以上颗粒物个数						H 														    R
//0X2F																														L                                 R
//——————————————————————————————————
//0X30							0.1 L空气中直径在1.0um以上颗粒物个数						H 																R
//0X31																														L 														    R
//0X31							0.1 L空气中直径在2.5um以上颗粒物个数						H														      R
//0X32							 																					 		L														      R
//0X33							0.1 L空气中直径在5.0um以上颗粒物个数						H														      R
//0X34							 																						 	L														      R
//0X35							0.1 L空气中直径在10um 以上颗粒物个数						H														      R
//0X36							 																							L														      R
//0X37                                                                                             
//0X38                                                                                             
//0X39                                                                                             
//0X3A                                                                                             
//0X3B                                                                                             
//0X3C                                                                                             
//0X3D                                                                                             
//0X3E                                                                                             
//0X3F

//起始均为0XAA   设备地址默认00
//————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
//读：	功能码:0x05      
//起始 	设备地址 	功能码 	寄存器起始地址	寄存器访问长度	CRC校验
//					
//返回	功能码:0x06  
//起始 	设备地址 	功能码 	返回有效字节数 	寄存器数据1 寄存器数据2 寄存器数据3 ... 寄存器数据N	CRC校验
//————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

//————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
//写：	功能码:0x07   
//起始 	设备地址 	功能码 	寄存器地址	寄存器写入数据	CRC校验

//返回	功能码:0x08
//起始 	设备地址 	功能码 	寄存器地址	寄存器写入数据	CRC校验
//————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

uint8_t RETURN_BUF[256];	

void GET_RES1_BUF(void){
	  do{//接收到最后一个字节 再延时10ms
			Res1++;
			HAL_Delay(1);
		}while(Res1<4);
		GET_UART1_NEWINFO();
		LED2=!LED2;			
		memset(Res1_Buf,0,256);
		Res1_Sign=0;	
	  Res1_Count=0;	
}

uint8_t GET_UART1_NEWINFO(){
	  union crcdata 
		{ 
			unsigned int word16;  
			unsigned char  byte[2]; 
			}crcnow; 
		
		uint16_t i;
		uint8_t order_start_addr;

	
		for(i=0;i<256;i++)if(Res1_Buf[i]==0xAA){order_start_addr=i;break;}//寻找命令头
		
		if(i>=250){return 0;}//找不到就退出
		
		if(Res1_Buf[order_start_addr+1]==MODBUS_ID){//命令针对本机
				
			if(Res1_Buf[order_start_addr+2]==0X05){//是读指令
				uint8_t READ_START_ADDR;
				uint8_t READ_LEN;				
				READ_START_ADDR=Res1_Buf[order_start_addr+3];
				READ_LEN=				Res1_Buf[order_start_addr+4];	
				
				if(READ_START_ADDR+READ_LEN>200)return 1;				
				
				memset(RETURN_BUF,0,256);
				RETURN_BUF[0]=0XAA;										//起始
				RETURN_BUF[1]=0X01;										//本机MODBUS ID
				RETURN_BUF[2]=0X06;										//功能码 读回复
					
				RETURN_BUF[3]=READ_LEN;				//发送数据长度 高字节				

				for(i=0;i<READ_LEN;i++){
					RETURN_BUF[4+i]=LINKAGE_BUF[READ_START_ADDR+i];	//发送数据	
				}
				
				crcnow.word16=chkcrc(RETURN_BUF,4+i); 	//计算CRC
								
				RETURN_BUF[4+i]=crcnow.byte[1];    //CRC 效验低字节在前 
				RETURN_BUF[5+i]=crcnow.byte[0]; 
				
				HAL_UART_Transmit(&huart1, RETURN_BUF, 6+i, 1000);			
			}
			if(Res1_Buf[order_start_addr+2]==0X07){//是写指令			
				uint16_t WIRTE_ADDR;	
				uint16_t WIRTE_DATA;

				WIRTE_ADDR=Res1_Buf[order_start_addr+3];	
				WIRTE_DATA=Res1_Buf[order_start_addr+4];

				if(WIRTE_ADDR>0X10)return 1;
				
				LINKAGE_BUF[WIRTE_ADDR]=WIRTE_DATA;
				
				memset(RETURN_BUF,0,256);
				RETURN_BUF[0]=0XAA;	
				RETURN_BUF[1]=0X01;						
				RETURN_BUF[2]=0X08;
				
				RETURN_BUF[3]=WIRTE_ADDR;		
				RETURN_BUF[4]=LINKAGE_BUF[WIRTE_ADDR];					

				crcnow.word16=chkcrc(RETURN_BUF,5); 
								
				RETURN_BUF[5]=crcnow.byte[1];    //CRC 效验低字节在前 
				RETURN_BUF[6]=crcnow.byte[0]; 
								
				HAL_UART_Transmit(&huart1, RETURN_BUF,7, 1000);
			}		
		}
		return 0;
}


unsigned int calccrc(unsigned char crcbuf,unsigned int crc){ 
	  unsigned char i;  
	  unsigned char chk; 
	  crc=crc^crcbuf;  
	  for(i=0;i<8;i++)
	  {   
	    chk=( unsigned char)(crc&1);
	    crc=crc>>1;  
	    crc=crc&0x7fff;  
   	  if(chk==1) 
	      crc=crc^0xa001; 
	    crc=crc&0xffff;  
	} 
	return crc;
	}
	
	
	
unsigned int chkcrc(unsigned char *buf,unsigned char len) {
	  unsigned char hi,lo; 
    unsigned int i;   
    unsigned int crc;   
    crc=0xFFFF;  
    for(i=0;i<len;i++)  
    {  
      crc=calccrc(*buf,crc); 
      buf++;
      } 
    hi=(unsigned char)(crc%256);  
    lo=(unsigned char)(crc/256); 
    crc=(((unsigned int)(hi))<<8)|lo; 
    return crc;   
} 


