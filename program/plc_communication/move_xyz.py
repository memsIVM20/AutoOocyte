import snap7
import struct
import snap7.client as client

# Constants for connection
PLC_IP = '192.168.0.1'
RACK = 0  # Rack number
SLOT = 1  # Slot number

plc = client.Client()
plc.connect(PLC_IP, RACK, SLOT)

def read_DB_number(db_number, data):
	reading = plc.db_read(db_number, data, 4)
	value = struct.unpack('>f', reading)  # big-endian
	return value[0]

def write_DB_number(db_number, start_address, value):
	plc.db_write(db_number, start_address, bytearray(struct.pack('>f', value)))

def writeBool_plc(db_number, start_offset, bit_offset, value):
    reading = plc.db_read(db_number, start_offset, 1)
    snap7.util.set_bool(reading, 0, bit_offset, value)
    plc.db_write(db_number, start_offset, reading)
    return None

def readBool_plc(db_number, start_offset, bit_offset):
    reading = plc.db_read(db_number, start_offset, 1)
    a = snap7.util.get_bool(reading, 0, bit_offset)
    return a

# Setup DB
STEPS_PER_REVOLUTION = 6400  # Number of steps per revolution

# DB Numbers in PLC
motor_x = 1
motor_y = 2
motor_z = 3

# Common Bool bit index
power = 0
move_Absolute = 1
move_Relative = 2
cong_tac = 3
jog_forward = 4
jog_back = 5
home = 6
is_done = 7

#start address = 19
done_flag = 1

# Common data byte index
distance_Abs = 2
distance_Rel = 6
velocity = 10
real_position = 14

def convert_mm_to_pulse(coordination):
    position_in_pulse = coordination / 5 * STEPS_PER_REVOLUTION
    return position_in_pulse

def move_coordination_motor_x(x_coord):
    coordination_x_pulse = convert_mm_to_pulse(x_coord)
    write_DB_number(motor_x, distance_Abs, coordination_x_pulse)
    writeBool_plc(motor_x, 0, move_Absolute, True)

def move_coordination_motor_y(y_coord):
    coordination_y_pulse = convert_mm_to_pulse(y_coord)
    write_DB_number(motor_y, distance_Abs, coordination_y_pulse)
    writeBool_plc(motor_y, 0, move_Absolute, True)

def move_coordination_motor_z(z_coord):
    coordination_z_pulse = convert_mm_to_pulse(z_coord)
    write_DB_number(motor_z, distance_Abs, coordination_z_pulse)
    writeBool_plc(motor_z, 0, move_Absolute, True)
    
def set_Home():
    writeBool_plc(motor_x, 0, home, 1)
    writeBool_plc(motor_y, 0, home, 1)
    writeBool_plc(motor_z, 0, home, 1)

def jog_forward_x():
    writeBool_plc(motor_x, 0, jog_forward, 1)
    
def jog_backward_x():
    writeBool_plc(motor_x, 0, jog_back, 1)
    
def jog_forward_y():
    writeBool_plc(motor_y, 0, jog_forward, 1)

def jog_backward_y():
    writeBool_plc(motor_y, 0, jog_back, 1)

def jog_forward_z():
    writeBool_plc(motor_z, 0, jog_forward, 1)

def jog_backward_z():
    writeBool_plc(motor_z, 0, jog_back, 1)
    
def stop_jogging_x():
    writeBool_plc(motor_x, 0, jog_forward, 0)
    writeBool_plc(motor_x, 0, jog_back, 0)
    
def stop_jogging_y():
    writeBool_plc(motor_y, 0, jog_forward, 0)
    writeBool_plc(motor_y, 0, jog_back, 0)
    
def stop_jogging_z():
    writeBool_plc(motor_z, 0, jog_forward, 0)
    writeBool_plc(motor_z, 0, jog_back, 0) 
    
def relative_x_forward():
    write_DB_number(motor_x, distance_Rel, 6)
    writeBool_plc(motor_x, 0, move_Relative, 1)
    writeBool_plc(motor_x, 0, move_Relative, 0)
    
def relative_y_forward():
    write_DB_number(motor_y, distance_Rel, 6)
    writeBool_plc(motor_y, 0, move_Relative, 1)
    writeBool_plc(motor_y, 0, move_Relative, 0)

def relative_z_forward():
    write_DB_number(motor_z, distance_Rel, 30)
    writeBool_plc(motor_z, 0, move_Relative, 1)
    writeBool_plc(motor_z, 0, move_Relative, 0)

def relative_x_backward():
    write_DB_number(motor_x, distance_Rel, -6)
    writeBool_plc(motor_x, 0, move_Relative, 1)
    writeBool_plc(motor_x, 0, move_Relative, 0)
    
def relative_y_backward():
    write_DB_number(motor_y, distance_Rel, -6)
    writeBool_plc(motor_y, 0, move_Relative, 1)
    writeBool_plc(motor_y, 0, move_Relative, 0)

def relative_z_backward():
    write_DB_number(motor_z, distance_Rel, -30)
    writeBool_plc(motor_z, 0, move_Relative, 1)
    writeBool_plc(motor_z, 0, move_Relative, 0)

def waiting_3_axis():
    while 1:
        x = readBool_plc(motor_x, 19, done_flag)
        y = readBool_plc(motor_y, 19, done_flag)
        z = readBool_plc(motor_z, 19, done_flag)
        if x == 1 & y == 1 & z == 1:
            writeBool_plc(motor_x, 19, done_flag, 0)
            writeBool_plc(motor_y, 19, done_flag, 0)
            writeBool_plc(motor_z, 19, done_flag, 0)
            break
        
def waiting_2_axis(motor_1, motor_2):
    while 1:
        motor_1_state = readBool_plc(motor_1, 19, done_flag)
        motor_2_state = readBool_plc(motor_2, 19, done_flag)
        if motor_1_state == 1 & motor_2_state == 1:
            motor_1_state = writeBool_plc(motor_1, 19, done_flag, 0)
            motor_2_state = writeBool_plc(motor_2, 19, done_flag, 0)
            break     

def waiting_1_axis(motor):
    while 1:
        motor_state = readBool_plc(motor, 19, done_flag)
        # print("HELP from ", motor, "state: ", motor_state)
        if motor_state == True:
            # reset bien trung gian
            motor_state = writeBool_plc(motor, 19, done_flag, 0)
            break
        
def read_current_position():
    x_pulse = read_DB_number(motor_x, real_position)
    y_pulse = read_DB_number(motor_y, real_position)
    z_pulse = read_DB_number(motor_z, real_position)
    # x_mm = convert_mm_to_pulse(x_pulse)
    # y_mm = convert_mm_to_pulse(y_pulse)
    # z_mm = convert_mm_to_pulse(z_pulse)
    # return x_mm, y_mm, z_mm
    x = x_pulse/6400 * 5
    y = y_pulse/6400 * 5
    z = z_pulse/6400 * 5
    
    return x, y, z

# set_Home()

# print(read_current_position())