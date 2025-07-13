#include "main.h" 
#include "action.h"


void MASTER_SCAN_ACTION(void){

	
	
	
	
	
	if(GET_ADC_FLAG){GET_NOISE_Value();GET_ADC_FLAG=0;}
	if(GET_DHT11_FLAG){GET_DHT11_Value();GET_DHT11_FLAG=0;}	
	if(GET_RAINSENSOR_FLAG){GET_RAIN_SENSOR();GET_RAINSENSOR_FLAG=0;}		
	
	
	
	
}






