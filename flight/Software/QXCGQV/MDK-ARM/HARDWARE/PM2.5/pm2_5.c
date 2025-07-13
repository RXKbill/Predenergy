
#include "pm2_5.h"
#include "main.h"


uint8_t GET_UART2_NEWINFO(void){
		uint8_t i,j;
	
		//42 4D 00 1C 00 04 00 06 00 06 00 04 00 06 00 06 04 02 01 1C 00 27 00 04 00 00 00 00 91 00 01 AA 	
		for(i=0;i<200;i++){
			if(Res2_Buf[i]==0x42&&Res2_Buf[i+1]==0x4D){
				if(Res2_Buf[i+2]==0x00&&Res2_Buf[i+3]==0x1C){
					for(j=0;j<24;j++){
						LINKAGE_BUF[0X20+j]=Res2_Buf[i+j+4];	
					}
				}
				break;				
			}
		}
		return 0;
}

//void USART2_Init(uint32_t bound)
//{  

////GPIO端口设置
//        GPIO_InitTypeDef GPIO_InitStructure;
//		USART_InitTypeDef USART_InitStructure;
//		NVIC_InitTypeDef NVIC_InitStructure;
//		 
//		RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA , ENABLE); //使能UART2所在GPIOA的时钟
//		RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART2, ENABLE); //使能串口的RCC时钟

//    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_3;              //设置USART2的RX接口是PA3
//    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;  
//    GPIO_Init(GPIOA, &GPIO_InitStructure); 

//    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2;              //设置USART2的TX接口是PA2
//    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
//    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;	       //复用推挽输出
//    GPIO_Init(GPIOA, &GPIO_InitStructure);  

//   //USART2 初始化设置
//		USART_InitStructure.USART_BaudRate = bound;           //一般设置为9600;
//		USART_InitStructure.USART_WordLength = USART_WordLength_8b;//字长为8位数据格式
//		USART_InitStructure.USART_StopBits = USART_StopBits_1;//一个停止位
//		USART_InitStructure.USART_Parity = USART_Parity_No;   //无奇偶校验位
//		USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;//无硬件数据流控制
//		USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;	//收发模式
//    USART_Init(USART2, &USART_InitStructure);             //初始化串口
//    USART_ITConfig(USART2, USART_IT_RXNE, ENABLE);        //开启ENABLE/关闭DISABLE中断
//    USART_Cmd(USART2, ENABLE);                            //使能串口 
//  
//	//Usart2 NVIC 配置
//	  NVIC_PriorityGroupConfig(NVIC_PriorityGroup_0);
//    NVIC_InitStructure.NVIC_IRQChannel = USART2_IRQn;
//		NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority=2 ;//抢占优先级
//		NVIC_InitStructure.NVIC_IRQChannelSubPriority = 1;		//优先级
//		NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;			  //IRQ通道使能
//		NVIC_Init(&NVIC_InitStructure);	                      //根据指定的参数初始化VIC寄存器  	

//}

//void send_code(void)
//{
//	int i = 0;
//	uint16_t send_buff[9] = {0xFF,0x01,0x86,0x00,0x00,0x00,0x00,0x00,0x79};
//	while(i<9)
//	{	
//		USART_SendData(USART2,send_buff[i]);
//		while( USART_GetFlagStatus(USART2,USART_FLAG_TC)!= SET); 
//		i++;
//	}
//}

//void USART2_IRQHandler(void)
//{	
//	static uint8_t seri_count=0;
//	uint16_t check_sum=0;         //校验和
//	uint16_t checkH,checkL;
//	uint8_t i,x;
//	static 	uint8_t flag;         //接收状态标记	  
//	if(USART_GetITStatus(USART2, USART_IT_RXNE) != RESET)  //接收中断
//	{
//		
//			if(USART_ReceiveData(USART2) == Data_Head1)        //接收到帧头
//			{
//				flag = 1;
//			}
//			
//			if(flag)
//			{	
//				USART_BUF[seri_count++] = USART_ReceiveData(USART2);  //存放一帧数据
//				if(seri_count == Data_Length)	
//				{
//					if(USART_BUF[0]==Data_Head1 && USART_BUF[1]==Data_Head2)
//					{
//						for( i = 0; i < Data_Length-2; i++)
//						{
//							check_sum += USART_BUF[i];        //计算校验和
// 					  }
//						checkL = check_sum & 0xFF;          //校验和低八位
//					
//            checkH = check_sum>>8;              //校验和高八位
//						
//						if((checkH == USART_BUF[Data_Length-2])&&(checkL == USART_BUF[Data_Length-1]))		
//						{
//						  PM25_data = USART_BUF[7] + USART_BUF[6]* 256;	
//						}	
//						seri_count = 0;
//						flag=0;
//					}
//					for( x = 0; x < 15; x++)
//					{
//						PM25_sum+=PM25_data;
//					}			
//					PM25_ave=PM25_sum/15;					
//				}	
//				PM25_sum=0;
//			}
//   } 
//} 
//	


//uint16_t Get_PM25_Average()
//{
//	uint8_t t;
//	uint16_t PM25_sum;
////	float PM25_ave;
//	
//	for(t=0;t<15;t++)
//	{
//	  PM25_sum+=PM25_data; 
//		delay_ms(5);
//	}
//	PM25_ave=PM25_sum/15;
//	return PM25_ave;
//}

/******************* (C) COPYRIGHT 2018 NTU *****END OF FILE****/

