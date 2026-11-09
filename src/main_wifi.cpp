#include <Arduino.h>
#include <SPI.h>
#include <string.h>
#include <WiFi.h>

// ================== WiFi / TCP 配置（请按你的环境修改，也可用 platformio 的 build_flags 传 -D）==================
#ifndef WIFI_SSID
#define WIFI_SSID "EEG_PC"
#endif
#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "90909090"
#endif
#define TCP_SERVER_PORT 3333
// 有 TCP 客户端时，USB 不打印每行 channel:（见 sendChannelLine）；此间隔打一条短心跳，避免误以为死机
#define TCP_SERIAL_HEARTBEAT_MS 3000u
// 无 TCP 客户端时：若 true 则数据行仍走 USB 串口（便于脱机调试）。
// 有 TCP 客户端时：数据行只发 WiFi，不再重复打串口，避免双路阻塞、115200 跟不上采样率导致整体变慢。
#ifndef WIFI_USE_SERIAL_DEBUG
#define WIFI_USE_SERIAL_DEBUG true
#endif

// ================== 宏定义区 ==================
#define CONFIG1_REG  0x95
#define CONFIG2_REG  0xC0
#define CONFIG3_REG  0xEC
#define CHnSET_REG   0x60
#define ENABLE_SRB1  0x20
#define BIAS_SENSP   0xFF
#define BIAS_SENSN   0xFF
#define LOFF_CONFIG   0x06
#define LOFF_SENSP    0xFF
#define LOFF_SENSN    0xFF
#define MODE_CONTINUOUS_READ   1
#define MODE_IMPEDANCE_MEASURE 2
#define MODE_SELF_TEST         3

#define CS_PIN    A3
#define SCLK_PIN  SCK
#define MOSI_PIN  MOSI
#define MISO_PIN  MISO
#define DRDY_PIN  A0
#define START_PIN A2
#define RESET_PIN A1

#define ADS1299_SPI_HZ 1000000
static const SPISettings kAds1299Spi(ADS1299_SPI_HZ, MSBFIRST, SPI_MODE1);

#define ADS1299_VREF_V   4.5
#define ADS1299_PGA_GAIN 24.0
#define ADS1299_2_POW_23 8388608.0

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
volatile bool dataReady = false;
int currentMode = MODE_CONTINUOUS_READ;
double channelDataBuffer[9];

static WiFiServer s_tcpServer(TCP_SERVER_PORT);
static WiFiClient s_streamClient;
static bool s_tcpServerStarted = false;
static uint32_t s_lastTcpHeartbeat = 0;

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
static bool connectWiFiBlocking(unsigned long timeoutMs);
static void ensureTcpServer();
static void maybeRetryWiFi();
static void acceptTcpClient();
static void printInfo(const char *msg);
static void handleModeCommand(char cmd);

static void sendChannelLine() {
  char buf[200];
  int n = snprintf(buf, sizeof(buf), "channel:%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\r\n",
                   channelDataBuffer[1], channelDataBuffer[2], channelDataBuffer[3], channelDataBuffer[4],
                   channelDataBuffer[5], channelDataBuffer[6], channelDataBuffer[7], channelDataBuffer[8]);
  if (n > 0 && n < (int)sizeof(buf)) {
    if (s_streamClient && s_streamClient.connected()) {
      // 整行一次性写出；之前用 availableForWrite 不足就丢整行，会造成有效刷新率远 <500Hz、曲线只更新在右侧
      s_streamClient.write((const uint8_t *)buf, (size_t)n);
    } else if (WIFI_USE_SERIAL_DEBUG) {
      // 未连 TCP 时才打串口；勿与 WiFi 同时满速打，否则 write/print 阻塞主循环
      Serial.print(buf);
    }
  }
}

// ================== setup ==================
void setup() {
  Serial.begin(115200);
  delay(200);

  pinMode(CS_PIN, OUTPUT);
  pinMode(SCLK_PIN, OUTPUT);
  pinMode(MOSI_PIN, OUTPUT);
  pinMode(MISO_PIN, INPUT);
  pinMode(DRDY_PIN, INPUT);
  pinMode(START_PIN, OUTPUT);
  pinMode(RESET_PIN, OUTPUT);

  digitalWrite(CS_PIN, HIGH);
  digitalWrite(START_PIN, LOW);
  digitalWrite(RESET_PIN, HIGH);
  delay(100);

  SPI.begin(SCLK_PIN, MISO_PIN, MOSI_PIN, CS_PIN);

  if (!connectWiFiBlocking(20000)) {
    printInfo("WiFi 首次连接失败，之后会在后台自动重试；ADS 仍先初始化。");
  }

  initADS1299();
  getDeviceID();
  printInfo("ADS1299 初始化完成（无线模式，TCP 流式数据）");

  attachInterrupt(digitalPinToInterrupt(DRDY_PIN), onDRDYInterrupt, FALLING);
  currentMode = MODE_CONTINUOUS_READ;
}

// ================== loop ==================
void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    ensureTcpServer();
  } else {
    s_tcpServerStarted = false;
    if (s_streamClient && s_streamClient.connected()) {
      s_streamClient.stop();
    }
    maybeRetryWiFi();
  }
  acceptTcpClient();

  while (Serial.available()) {
    handleModeCommand((char)Serial.read());
  }
  if (s_streamClient && s_streamClient.connected() && s_streamClient.available()) {
    handleModeCommand((char)s_streamClient.read());
  }

  if (dataReady) {
    dataReady = false;
    readData();
    sendChannelLine();
  }

  // 有 TCP 时 USB 不刷 channel: 行，短心跳表示主循环/ADS 仍运行
  if (s_streamClient && s_streamClient.connected() &&
      (millis() - s_lastTcpHeartbeat) >= TCP_SERIAL_HEARTBEAT_MS) {
    s_lastTcpHeartbeat = millis();
    printInfo("[TCP 心跳] 有客户端时 data 只走 WiFi，USB 不刷 channel: 行。");
  }
}

// ------------------ WiFi / TCP ------------------
static bool connectWiFiBlocking(unsigned long timeoutMs) {
  if (strcmp(WIFI_SSID, "YOUR_WIFI_SSID") == 0) {
    printInfo("请先在 main_wifi.cpp 中设置 WIFI_SSID / WIFI_PASSWORD，或通过 build_flags 传入。");
  }
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < timeoutMs) {
    delay(200);
  }
  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }
  ensureTcpServer();
  IPAddress ip = WiFi.localIP();
  char line[96];
  snprintf(line, sizeof(line), "已连接 WiFi，IP: %s  监听 TCP 端口: %d",
           ip.toString().c_str(), (int)TCP_SERVER_PORT);
  printInfo(line);
  printInfo("在电脑上以 TCP 客户端连接上述 IP:端口；数据行格式与串口版相同: channel:...");
  return true;
}

static void ensureTcpServer() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }
  WiFi.setSleep(false);
  if (!s_tcpServerStarted) {
    s_tcpServer.begin();
    s_tcpServerStarted = true;
  }
}

static void maybeRetryWiFi() {
  static uint32_t s_next = 0;
  uint32_t now = millis();
  if (now < s_next) {
    return;
  }
  s_next = now + 5000;
  WiFi.disconnect();
  delay(50);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

static void acceptTcpClient() {
  if (s_streamClient && s_streamClient.connected()) {
    return;
  }
  WiFiClient incoming = s_tcpServer.available();
  if (incoming) {
    s_streamClient = incoming;
    s_streamClient.setNoDelay(true);
    s_lastTcpHeartbeat = millis();
    printInfo("已有客户端连接 TCP。提示：有 TCP 时不再通过 USB 打印每行 channel:，数据只在 WiFi 上，属于正常；下面会周期性打心跳行。");
  }
}

static void printInfo(const char *msg) {
  Serial.println(msg);
}

static void handleModeCommand(char cmd) {
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

// ================== 中断服务函数 ==================
void IRAM_ATTR onDRDYInterrupt() {
  dataReady = true;
}

// ================== 模式配置（与 main.cpp 相同）==================
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
  printInfo("阻抗测量模式已启用（注意需解调导联频率信号）");
}

void startSelfTestMode() {
  sendCommand(RESET);
  delay(100);
  sendCommand(SDATAC);
  writeRegister(0x01, 0x95);
  writeRegister(0x02, 0xD1);
  writeRegister(0x03, CONFIG3_REG);
  for (int i = 0x05; i <= 0x0C; i++) writeRegister(i, 0x65);
  sendCommand(START);
  sendCommand(RDATAC);
  printInfo("自检模式已启用");
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
  writeRegister(0x04, 0x00);
  writeRegister(0x0F, 0x00);
  writeRegister(0x10, 0x00);
  sendCommand(START);
  sendCommand(RDATAC);
}

// ================== 数据读取 ==================
void readData() {
  byte data[27];
  delayMicroseconds(2);
  SPI.beginTransaction(kAds1299Spi);
  digitalWrite(CS_PIN, LOW);
  for (int i = 0; i < 27; i++) data[i] = SPI.transfer(0x00);
  digitalWrite(CS_PIN, HIGH);
  SPI.endTransaction();
  convertData(data, channelDataBuffer);
}

void convertData(byte *data, double *channelData) {
  long statusValue = ((long)data[0] << 16) | ((long)data[1] << 8) | data[2];
  channelData[0] = (double)statusValue;
  const double scale = ADS1299_VREF_V / (ADS1299_PGA_GAIN * ADS1299_2_POW_23);
  for (int i = 0; i < 8; i++) {
    long raw = ((long)data[3 * i + 3] << 16) | ((long)data[3 * i + 4] << 8) | data[3 * i + 5];
    if (raw & 0x800000) raw |= 0xFF000000;
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
  if (WIFI_USE_SERIAL_DEBUG) {
    Serial.print("Device ID: 0b");
    Serial.println(id, BIN);
  }
}
