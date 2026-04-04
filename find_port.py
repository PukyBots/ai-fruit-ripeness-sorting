import serial.tools.list_ports

ports = list(serial.tools.list_ports.comports())

if not ports:
    print("No USB devices found.")
    print("Make sure arm is plugged in and powered on.")
else:
    print(f"Found {len(ports)} USB device(s):")
    for p in ports:
        print(f"  Port: {p.device}")
        print(f"  Name: {p.description}")
        print(f"  Hwid: {p.hwid}")
        print()

print("Use the port shown above in arm_control.py")
print("Example: SERIAL_PORT = 'COM3'")