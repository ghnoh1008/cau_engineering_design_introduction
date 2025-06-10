#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <RTClib.h>
#include <SD.h>

// 핀 정의
const int sensorPin1 = A0;
const int sensorPin2 = A1;
const int chipSelect = 10;

// 거리 및 센서 변수
float cm1, cm2;
int Sensor_val1, Sensor_val2;

// 이상 탐지 기준값
const float movementThreshold = 1.5   // 정적 판단 기준(cm)
const float noiseFloor = 0.5;          // 노이즈 제거 기준(cm)
const int stableLimit = 10;            // 연속 정적 판단 횟수
const float maxDrift = 20.0;            // 누적 거리 변화 한계(cm)

// 상태 변수
float prev_cm1 = 0, prev_cm2 = 0;
float firstStable_cm1 = 0, firstStable_cm2 = 0;
float totalDrift = 0;
int stableCount = 0;

// 시간 제어 변수
unsigned long lastCheckTime = 0;
unsigned long checkInterval = 1000;

// LCD, RTC, SD 객체
LiquidCrystal_I2C lcd(0x27, 16, 2);
RTC_DS3231 rtc;
File logFile;

void setup() {
  Serial.begin(9600);
  pinMode(10, OUTPUT); // SD 카드 SPI 통신용 필요

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("System Initializing");

  // RTC 초기화
  if (!rtc.begin()) {
    Serial.println("Couldn't find RTC");
    lcd.setCursor(0, 1);
    lcd.print("RTC Fail");
    while (1);
  }

  // SD 카드 초기화
  if (!SD.begin(chipSelect)) {
    Serial.println("SD init fail!");
    lcd.setCursor(0, 1);
    lcd.print("SD Fail");
  } else {
    Serial.println("SD init success.");
  }

  delay(2000);
  lcd.clear();
}

void loop() {
  unsigned long nowMillis = millis();
  if (nowMillis - lastCheckTime < checkInterval) return;
  lastCheckTime = nowMillis;

  // 센서 값 읽기 및 거리 변환
  Sensor_val1 = map(analogRead(sensorPin1), 0, 1023, 0, 5000);
  Sensor_val2 = map(analogRead(sensorPin2), 0, 1023, 0, 5000);
  cm1 = (24.61 / (Sensor_val1 - 0.1696)) * 1000;
  cm2 = (24.61 / (Sensor_val2 - 0.1696)) * 1000;

  // 유효 범위 필터링
  if (cm1 < 8 || cm1 > 30) cm1 = prev_cm1;
  if (cm2 < 8 || cm2 > 30) cm2 = prev_cm2;

  // 변화량 계산
  float delta1 = abs(cm1 - prev_cm1);
  float delta2 = abs(cm2 - prev_cm2);

  Serial.print("cm1: ");
  Serial.print(cm1, 1);  // 소수점 1자리까지 출력
  Serial.print(" cm\t");

  Serial.print("cm2: ");
  Serial.print(cm2, 1);  // 소수점 1자리까지 출력
  Serial.println(" cm");


  // 노이즈 제거
  if (delta1 < noiseFloor) delta1 = 0;
  if (delta2 < noiseFloor) delta2 = 0;

  bool isStable = (delta1 < movementThreshold) && (delta2 < movementThreshold);

  if (isStable) {
    stableCount++;
    if (stableCount == 1) {
      // 정적 상태 시작값 기록
      firstStable_cm1 = cm1;
      firstStable_cm2 = cm2;
      totalDrift = 0;
    } else {
      // 누적 거리 변화 측정
      totalDrift += abs(cm1 - firstStable_cm1);
      totalDrift += abs(cm2 - firstStable_cm2);
    }
  } else {
    stableCount = 0;
    totalDrift = 0;
  }

  // 이상 상태 탐지
  if (stableCount >= stableLimit && totalDrift < maxDrift) {
    DateTime now = rtc.now();

    // LCD 경고 출력
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("abnormal condition");
    lcd.setCursor(0, 1);
    lcd.print(now.timestamp(DateTime::TIMESTAMP_TIME));

    // SD 카드 로그 기록
    logFile = SD.open("log.csv", FILE_WRITE);
    if (logFile) {
      logFile.print(now.timestamp(DateTime::TIMESTAMP_FULL));
      logFile.println(",abnormal condition");
      logFile.close();
      Serial.println("Log written.");
    } else {
      Serial.println("Couldn't open log file");
    }

    delay(3000);
    lcd.clear();
  }

  // 상태 업데이트
  prev_cm1 = cm1;
  prev_cm2 = cm2;
}
