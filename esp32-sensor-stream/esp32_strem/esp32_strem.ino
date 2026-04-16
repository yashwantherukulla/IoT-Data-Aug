#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <Wire.h>

// ─── WiFi & Server Config ────────────────────────────────────
const char* WIFI_SSID = "GEMINI";
const char* WIFI_PASSWORD = "123456789";
const char* SERVER_URL = "http://10.192.5.194:8000/api/sensor";

const String DEVICE_ID = "ESP32-NODE-1";

// ─── Hardware ────────────────────────────────────────────────
const int MPU_ADDR = 0x68;
const int LED_PIN = 18;    // D18 – Network Status
const int LED_PIN2 = 19;   // D19 – Activity Blink

// ─── Batching ────────────────────────────────────────────────
const int BATCH_SIZE = 10; 
float accBatch[BATCH_SIZE][3];
float gyroBatch[BATCH_SIZE][3];
int batchIndex = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  
  pinMode(LED_PIN, OUTPUT);
  pinMode(LED_PIN2, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(LED_PIN2, LOW);

  // Connect WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  digitalWrite(LED_PIN, HIGH); // Turn on D18 to show WiFi connected

  initMPU6050();
}

void loop() {
  readMPUSample();
  
  batchIndex++;
  if (batchIndex >= BATCH_SIZE) {
    sendData();
    batchIndex = 0;
  }
  
  delay(20); // ~50Hz sampling rate
}

void initMPU6050() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);
  Wire.write(0x18); // Accelerometer ±16 g
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B);
  Wire.write(0x18); // Gyroscope ±2000°/s
  Wire.endTransmission(true);
}

void readMPUSample() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);

  if (Wire.available() < 14) return;

  int16_t rawAcX = Wire.read() << 8 | Wire.read();
  int16_t rawAcY = Wire.read() << 8 | Wire.read();
  int16_t rawAcZ = Wire.read() << 8 | Wire.read();
  Wire.read(); Wire.read(); // skip temp
  int16_t rawGyX = Wire.read() << 8 | Wire.read();
  int16_t rawGyY = Wire.read() << 8 | Wire.read();
  int16_t rawGyZ = Wire.read() << 8 | Wire.read();

  accBatch[batchIndex][0] = (rawAcX / 2048.0f) * 9.81f;
  accBatch[batchIndex][1] = (rawAcY / 2048.0f) * 9.81f;
  accBatch[batchIndex][2] = (rawAcZ / 2048.0f) * 9.81f;

  gyroBatch[batchIndex][0] = rawGyX / 16.4f;
  gyroBatch[batchIndex][1] = rawGyY / 16.4f;
  gyroBatch[batchIndex][2] = rawGyZ / 16.4f;
}

void sendData() {
  digitalWrite(LED_PIN2, HIGH); // Blink D19 when sending
  
  WiFiClient client;
  HTTPClient http;
  http.begin(client, SERVER_URL);
  http.addHeader("Content-Type", "application/json");

  StaticJsonDocument<4096> doc;
  doc["device_id"] = DEVICE_ID;

  JsonArray acc = doc.createNestedArray("acc");
  JsonArray gyro = doc.createNestedArray("gyro");

  for (int i = 0; i < BATCH_SIZE; i++) {
    JsonArray a = acc.createNestedArray();
    a.add(accBatch[i][0]);
    a.add(accBatch[i][1]);
    a.add(accBatch[i][2]);
    
    JsonArray g = gyro.createNestedArray();
    g.add(gyroBatch[i][0]);
    g.add(gyroBatch[i][1]);
    g.add(gyroBatch[i][2]);
  }

  String payload;
  serializeJson(doc, payload);

  int httpCode = http.POST(payload);
  if (httpCode > 0) {
    Serial.println("Batch sent successfully.");
  } else {
    Serial.println("HTTP request failed.");
  }
  
  http.end();
  digitalWrite(LED_PIN2, LOW);
}