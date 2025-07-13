#include "main.h" 
#include "key.h"






//按键初始化 在GPIO_INIT中已经完成
void KEY_INIT(void){
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOB_CLK_ENABLE();
	
  /*Configure GPIO pins : KEY2_PB12_Pin KEY3_PB13_Pin KEY4_PB14_Pin KEY5_PB15_Pin */
  GPIO_InitStruct.Pin = GPIO_PIN_3|GPIO_PIN_4;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI3_IRQn, 1, 1);
  HAL_NVIC_EnableIRQ(EXTI3_IRQn);

  HAL_NVIC_SetPriority(EXTI4_IRQn, 1, 1);
  HAL_NVIC_EnableIRQ(EXTI4_IRQn);


}





void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin){
		switch(GPIO_Pin){
        case GPIO_PIN_8: //key1 

            break;

        case GPIO_PIN_12:


           break;						
						
				case GPIO_PIN_13:

            break;	

				case GPIO_PIN_14://key4

            break;	

				case GPIO_PIN_15://key5
		
            break;								
    }	
}


















