#include <MsTimer2.h>
#include <SoftwareSerial.h>
SoftwareSerial serial(2, 3);
int count = 0;
int dat_len = 30, dat_cnt = 2;
byte ch;
uint8_t temp[33];
uint16_t jiaoyan = 0x0000;
struct data {
  uint16_t bpm1p0;
  uint16_t bpm2p5;
  uint16_t bpm10;
  uint16_t pm1p0;
  uint16_t pm2p5;
  uint16_t pm10;
  uint16_t pm0p3u;
  uint16_t pm0p5u;
  uint16_t pm1p0u;
  uint16_t pm2p5u;
  uint16_t pm5p0u;
  uint16_t pm10u;
};
data pm;
void onTimer()
{
  while (serial.available() > 0)
  {
    ch = byte(serial.read());
    if (count == 0 && ch == 0x42)
    {
      count = 1;
      temp[0] = ch;
    }
    else if (count == 1 && ch == 0x4D)
    {
      count = 2;
      temp[1] = ch;
    }
    else if (count == 2 && dat_len > 0)
    {
      dat_len--;
      temp[dat_cnt++] = ch;
      if (dat_len == 0) count = 3;
    }
    else if (count == 3)
    {
      MsTimer2::stop();
      count = 0;
      dat_len = 30;
      dat_cnt = 2;
      DataGive();
      MsTimer2::start();
    }
  }
}
void DataGive()
{
  for (int i = 0; i < 30; i ++)
  {
    jiaoyan += temp[i] ;
  }
  if (jiaoyan != ((temp[30] << 8)  | temp[31]))
  {
    jiaoyan = 0x0000;
    Serial.println("????!");
    return;
  }
  jiaoyan = 0x0000;
  pm.bpm1p0 = (temp[4] << 8) | temp[5];
  pm.bpm2p5 = (temp[6] << 8) | temp[7];
  pm.bpm10  = (temp[8] << 8) | temp[9];
  pm.pm1p0  = (temp[10] << 8) | temp[11];
  pm.pm2p5  = (temp[12] << 8) | temp[13];
  pm.pm10   = (temp[14] << 8) | temp[15];
  pm.pm0p3u = (temp[16] << 8) | temp[17];
  pm.pm0p5u = (temp[18] << 8) | temp[19];
  pm.pm1p0u = (temp[20] << 8) | temp[21];
  pm.pm2p5u = (temp[22] << 8) | temp[23];
  pm.pm5p0u = (temp[24] << 8) | temp[25];
  pm.pm10u  = (temp[26] << 8) | temp[27];
  Serial.print("CF=1,?????? ");
  Serial.print("PM1.0?:");
  Serial.print(pm.bpm1p0);
  Serial.print(" PM2.5?:");
  Serial.print(pm.bpm2p5);
  Serial.print(" PM10?:");
  Serial.println(pm.bpm10);
  
  Serial.print("??????      ");
  Serial.print("PM1.0?:");
  Serial.print(pm.pm1p0);
  Serial.print(" PM2.5?:");
  Serial.print(pm.pm2p5);
  Serial.print(" PM10?:");
  Serial.println(pm.pm10);

  Serial.print("0.1 ??????? 0.3um ??? ?????:");
  Serial.println(pm.pm0p3u);
  Serial.print("0.1 ??????? 0.5um ??? ?????:");
  Serial.println(pm.pm0p5u);
  Serial.print("0.1 ??????? 1.0um ??? ?????:");
  Serial.println(pm.pm1p0u);
  Serial.print("0.1 ??????? 2.5um ??? ?????:");
  Serial.println(pm.pm2p5u);
  Serial.print("0.1 ??????? 5.0um ??? ?????:");
  Serial.println(pm.pm5p0u);
  Serial.print("0.1 ??????? 10 um ??? ?????:");
  Serial.println(pm.pm10u); 
}
void setup()
{
  Serial.begin(9600);
  serial.begin(9600);
  MsTimer2::set(2000, onTimer);
  MsTimer2::start();
}

void loop()
{
}
