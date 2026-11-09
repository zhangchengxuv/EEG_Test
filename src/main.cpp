#include <Arduino.h>
#include <SPI.h>

// ================== 宏定义区 ==================
// ---------- 采样率配置 ----------
#define CONFIG1_REG  0x95   // 单片模式，500 SPS
#define CONFIG2_REG  0xC0   // 常规采样，无测试信号
#define CONFIG3_REG  0xEC   // 内部参考、BIAS 缓冲开启

// ---------- 通道配置 ----------
#define CHnSET_REG   0x60   // 增益 24x，普通电极输入
#define ENABLE_SRB1  0x20   // MISC1: 启用 SRB1
#define BIAS_SENSP   0xFF   // BIAS 正端全开
#define BIAS_SENSN   0xFF   // BIAS 负端全开

// ---------- 阻抗测量配置 ----------
#define LOFF_CONFIG   0x06  // 选择 6nA，FLEAD_OFF=10(31.2Hz)
#define LOFF_SENSP    0xFF  // 开启所有通道正端注入
#define LOFF_SENSN    0xFF  // 开启所有通道负端注入

// ---------- 模式选择 ----------
#define MODE_CONTINUOUS_READ   1
#define MODE_IMPEDANCE_MEASURE 2
#define MODE_SELF_TEST         3

// ================== 引脚定义 ==================
#define CS_PIN    A3
#define SCLK_PIN  SCK
#define MOSI_PIN  MOSI
#define MISO_PIN  MISO
#define DRDY_PIN  A0
#define START_PIN A2
#define RESET_PIN A1

// ADS1299：SPI ≤2MHz（手册）；ESP32 用显式频率，勿用 AVR 的 SPI_CLOCK_DIVn）
#define ADS1299_SPI_HZ 1000000
static const SPISettings kAds1299Spi(ADS1299_SPI_HZ, MSBFIRST, SPI_MODE1);

// 换算（手册）：V_in = Code × Vref / (Gain × 2^23)。此为差分电压 (Vin+ − Vin−)，
// 非对地“绝对电压”。EEG 常见接法（负端/SRB1 作参考）下，表示电极相对 SRB1 的相对电位。
#define ADS1299_VREF_V   4.5     // 内部参考典型值 (V)，以 CONFIG3 实际参考源为准
#define ADS1299_PGA_GAIN 24.0    // 须与 CHnSET 增益一致
#define ADS1299_2_POW_23 8388608.0  // 2^23

// ================== 命令定义 ==================
#define WAKEUP  0x02
#define STANDBY 0x04
#define RESET   0x06
#define START   0x08
#define STOP    0x0A
#define RDATAC  0x10
#define SDATAC  0x11
#define RDATA   0x12
#define RREG    0x20
#define WREG    0x40

// ================== 全局变量 ==================
volatile bool dataReady = false; // DRDY 中断标志
int currentMode = MODE_CONTINUOUS_READ;
double channelDataBuffer[9]; // [0]=STATUS；[1..8]=各通道差分相对电位 (V)，相对 SRB1 参考，非绝对电位

// ================== 函数声明 ==================
void IRAM_ATTR onDRDYInterrupt();
void initADS1299();
void startContinuousReadMode();
void startImpedanceMeasurementMode();
void startSelfTestMode();
void readData();
void convertData(byte *data, double *channelData);
void sendCommand(byte cmd);
void writeRegister(byte reg, byte value);
byte readRegister(byte reg);
void getDeviceID();

// ================== setup ==================
void setup() {
  Serial.begin(115200);

  // 初始化引脚
  pinMode(CS_PIN, OUTPUT);
  pinMode(SCLK_PIN, OUTPUT);
  pinMode(MOSI_PIN, OUTPUT);
  pinMode(MISO_PIN, INPUT);
  pinMode(DRDY_PIN, INPUT);
  pinMode(START_PIN, OUTPUT);
  pinMode(RESET_PIN, OUTPUT);

  digitalWrite(CS_PIN, HIGH);  // nCS 低有效，空闲保持未选中
  digitalWrite(START_PIN, LOW);
  digitalWrite(RESET_PIN, HIGH);
  delay(100);

  // 初始化 SPI（频率在每次传输时用 kAds1299Spi 包裹）
  SPI.begin(SCLK_PIN, MISO_PIN, MOSI_PIN, CS_PIN);

  // 初始化 ADS1299
  initADS1299();
  getDeviceID();
  Serial.println("ADS1299 初始化完成");

  // 配置外部中断（DRDY 下降沿触发）
  attachInterrupt(digitalPinToInterrupt(DRDY_PIN), onDRDYInterrupt, FALLING);

  currentMode = MODE_CONTINUOUS_READ;
}

// ================== loop ==================
void loop() {
  // 串口命令切换模式
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == '1') {
      currentMode = MODE_CONTINUOUS_READ;
      startContinuousReadMode();
    } else if (cmd == '2') {
      currentMode = MODE_IMPEDANCE_MEASURE;
      startImpedanceMeasurementMode();
    } else if (cmd == '3') {
      currentMode = MODE_SELF_TEST;
      startSelfTestMode();
    }
  }

  // DRDY 仅在中断里置位；SPI 在主循环执行（勿在中断里读 SPI）
  if (dataReady) {
    dataReady = false;
    readData();
    // 串口 CH1..CH8 顺序对应电极：PO7, PO3, POz, PO4, PO8, O1, Oz, O2（与 PC 端 MNE 通道名一致）
    Serial.print("channel:");
    for (int i = 1; i <= 8; i++) {
      Serial.print(channelDataBuffer[i], 6); // 差分相对电位 (V)，非对地绝对电压
      if (i < 8) Serial.print(",");
    }
    Serial.println();
  }
}

// ================== 中断服务函数 ==================
void IRAM_ATTR onDRDYInterrupt() {
  dataReady = true;
}

// ================== 模式配置 ==================
void startContinuousReadMode() {
  sendCommand(RESET);
  delay(100);
  sendCommand(SDATAC);
  writeRegister(0x01, CONFIG1_REG);
  writeRegister(0x02, CONFIG2_REG);
  writeRegister(0x03, CONFIG3_REG);
  for (int i = 0x05; i <= 0x0C; i++) writeRegister(i, CHnSET_REG);
  writeRegister(0x0D, BIAS_SENSP);
  writeRegister(0x0E, BIAS_SENSN);
  writeRegister(0x15, ENABLE_SRB1);
  sendCommand(START);
  sendCommand(RDATAC);
}

void startImpedanceMeasurementMode() {
  sendCommand(RESET);
  delay(100);
  sendCommand(SDATAC);
  writeRegister(0x04, LOFF_CONFIG);
  writeRegister(0x0F, LOFF_SENSP);
  writeRegister(0x10, LOFF_SENSN);
  for (int i = 0x05; i <= 0x0C; i++) writeRegister(i, CHnSET_REG);
  sendCommand(START);
  sendCommand(RDATAC);
  Serial.println("阻抗测量模式已启用（注意需解调导联频率信号）");
}

void startSelfTestMode() {
  sendCommand(RESET);
  delay(100);
  sendCommand(SDATAC);
  writeRegister(0x01, 0x95); // 500SPS
  writeRegister(0x02, 0xD1); // 开启测试信号
  writeRegister(0x03, CONFIG3_REG);
  for (int i = 0x05; i <= 0x0C; i++) writeRegister(i, 0x65); // 测试信号输入
  sendCommand(START);
  sendCommand(RDATAC);
  Serial.println("自检模式已启用");
}

// ================== 初始化 ==================
void initADS1299() {
  sendCommand(RESET);
  delay(100);
  sendCommand(SDATAC);

  writeRegister(0x01, CONFIG1_REG);
  writeRegister(0x02, CONFIG2_REG);
  writeRegister(0x03, CONFIG3_REG);
  for (int i = 0x05; i <= 0x0C; i++) writeRegister(i, CHnSET_REG);
  writeRegister(0x0D, BIAS_SENSP);
  writeRegister(0x0E, BIAS_SENSN);
  writeRegister(0x15, ENABLE_SRB1);

  // 默认关闭导联检测
  writeRegister(0x04, 0x00);
  writeRegister(0x0F, 0x00);
  writeRegister(0x10, 0x00);

  sendCommand(START);
  sendCommand(RDATAC);
}

// ================== 数据读取 ==================
void readData() {
  byte data[27];
  // DRDY 下降沿后留短暂建立时间再拉 CS / 读 SPI（手册时序余量）
  delayMicroseconds(2);
  SPI.beginTransaction(kAds1299Spi);
  digitalWrite(CS_PIN, LOW);
  for (int i = 0; i < 27; i++) data[i] = SPI.transfer(0x00);
  digitalWrite(CS_PIN, HIGH);
  SPI.endTransaction();
  convertData(data, channelDataBuffer);
}

void convertData(byte *data, double *channelData) {
  // 解析 STATUS
  long statusValue = ((long)data[0] << 16) | ((long)data[1] << 8) | data[2];
  channelData[0] = (double)statusValue;

  // V_in = Code × Vref / (Gain × 2^23) → 差分 (Vin+ − Vin−)；EEG 下为相对 SRB1 的电位差
  const double scale = ADS1299_VREF_V / (ADS1299_PGA_GAIN * ADS1299_2_POW_23);
  for (int i = 0; i < 8; i++) {
    long raw = ((long)data[3*i+3] << 16) | ((long)data[3*i+4] << 8) | data[3*i+5];
    if (raw & 0x800000) raw |= 0xFF000000; // 符号扩展
    channelData[i + 1] = (double)raw * scale;
  }
}

// ================== 底层 SPI 操作 ==================
void sendCommand(byte cmd) {
  SPI.beginTransaction(kAds1299Spi);
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(cmd);
  digitalWrite(CS_PIN, HIGH);
  SPI.endTransaction();
}

void writeRegister(byte reg, byte value) {
  SPI.beginTransaction(kAds1299Spi);
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(WREG | reg);
  SPI.transfer(0x00);
  SPI.transfer(value);
  digitalWrite(CS_PIN, HIGH);
  SPI.endTransaction();
}

byte readRegister(byte reg) {
  SPI.beginTransaction(kAds1299Spi);
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(RREG | reg);
  SPI.transfer(0x00);
  byte val = SPI.transfer(0x00);
  digitalWrite(CS_PIN, HIGH);
  SPI.endTransaction();
  return val;
}

void getDeviceID() {
  SPI.beginTransaction(kAds1299Spi);
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(SDATAC);
  SPI.transfer(RREG | 0x00);
  SPI.transfer(0x00);
  byte id = SPI.transfer(0x00);
  digitalWrite(CS_PIN, HIGH);
  SPI.endTransaction();
  Serial.print("Device ID: 0b");
  Serial.println(id, BIN);
}