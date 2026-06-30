// CNC Shield V3 - X Axis

#define STEP_PIN 2
#define DIR_PIN 5
#define ENABLE_PIN 8

void setup() {

  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);

  // Enable motor driver (LOW = Enable)
  digitalWrite(ENABLE_PIN, LOW);
}

void loop() {

  // Clockwise
  digitalWrite(DIR_PIN, HIGH);

  for (int i = 0; i < 200; i++) {   // 200 steps = 1 revolution (1.8° motor)
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(800);

    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(800);
  }

  delay(1000);

  // Counter-clockwise
  digitalWrite(DIR_PIN, LOW);

  for (int i = 0; i < 200; i++) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(800);

    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(800);
  }

  delay(1000);
}
