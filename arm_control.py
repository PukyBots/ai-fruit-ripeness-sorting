import time

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

MOCK_MODE = True

SERIAL_PORT = 'COM3'
BAUD_RATE = 9600


class SCARAArm:
    def __init__(self):
        self.connected = False
        self.ser = None

        if MOCK_MODE:
            print("[ARM] Mock mode ON — simulating arm movements")
            print("[ARM] Set MOCK_MODE=False when real arm connected")
            self.connected = True
            return

        
        if not SERIAL_AVAILABLE:
            print("[ARM] pyserial not installed — run: pip install pyserial")
            return

        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
            time.sleep(2)
            self.connected = True
            print(f"[ARM] Connected on {SERIAL_PORT}")
        except Exception as e:
            print(f"[ARM] Connection failed: {e}")
            print(f"[ARM] Check port in Device Manager")

    def send(self, cmd):
        if MOCK_MODE:
            print(f"[ARM SIM] Sending: {cmd}")
            time.sleep(0.5)
            return

        if self.ser and self.ser.is_open:
            self.ser.write((cmd + '\n').encode())
            time.sleep(0.3)

    def home(self):
        print("[ARM] Moving to HOME position")
        self.send("G28")
        time.sleep(2)

    def pick(self):
        print("[ARM] PICKING fruit")
        self.send("PICK")
        time.sleep(1.5)

    def place_zone_a(self):
        print("[ARM] Placing in ZONE A (Ripe)")
        self.send("ZONE_A")
        time.sleep(2)

    def place_zone_b(self):
        print("[ARM] Placing in ZONE B (Unripe/Defective)")
        self.send("ZONE_B")
        time.sleep(2)

    def open_gripper(self):
        print("[ARM] Opening gripper")
        self.send("GRIP_OPEN")
        time.sleep(1)

    def close_gripper(self):
        print("[ARM] Closing gripper")
        self.send("GRIP_CLOSE")
        time.sleep(1)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[ARM] Serial connection closed")

if __name__ == "__main__":
    print("=" * 40)
    print("SCARA Arm Test")
    print("=" * 40)

    arm = SCARAArm()

    if arm.connected:
        print("Running movement sequence...")
        arm.home()
        time.sleep(1)
        arm.pick()
        arm.place_zone_a()
        arm.home()
        time.sleep(1)
        arm.pick()
        arm.place_zone_b()
        arm.home()
        arm.close()
        print("Test complete!")
    else:
        print("Arm not connected.")