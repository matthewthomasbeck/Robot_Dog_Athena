# Old Model Shape vs Isaac Lab Model Shape - Migration Guide

## Overview

This document compares your current custom model (97-dimensional state vector) with the Isaac Lab model (48-dimensional observation vector) to help you migrate smoothly.

---

## Model Dimensions

| Model | Dimension | Description |
|-------|-----------|-------------|
| **Old Model** | **97 values** | Custom RL model with temporal history |
| **Isaac Lab Model** | **48 values** | Standard Isaac Lab locomotion model |

---

## Old Model Structure (97 values)

### Breakdown:

1. **Historical Joint Angles** (60 values)
   - 5 timesteps × 12 joints = 60 values
   - Each timestep: [FL_hip, FL_upper, FL_lower, FR_hip, FR_upper, FR_lower, BL_hip, BL_upper, BL_lower, BR_hip, BR_upper, BR_lower]
   - Values: Normalized model outputs in range [-1, 1]
   - Stored in: `config.PREVIOUS_POSITIONS[0]` (deque with maxlen=5)

2. **Historical Orientation Data** (30 values)
   - 5 timesteps × 6 orientation values = 30 values
   - Each timestep: [shift, move, translate, yaw, roll, pitch]
   - Values: Normalized to [-1, 1] range
   - Stored in: `config.PREVIOUS_ORIENTATIONS[0]` (deque with maxlen=5)

3. **Command Encoding** (6 values)
   - One-hot encoding: [w, s, a, d, arrowleft, arrowright]
   - Values: 0.0 or 1.0

4. **Intensity** (1 value)
   - Normalized: `(intensity - 5.5) / 4.5`
   - Range: Approximately [-1, 1] (depending on input range)

**Total: 60 + 30 + 6 + 1 = 97 values**

---

## Isaac Lab Model Structure (48 values)

### Breakdown:

1. **base_lin_vel** (3 values)
   - Linear velocity in world frame: [vx, vy, vz]
   - Units: **m/s**
   - Frame: World frame
   - Noise during training: Uniform [-0.1, 0.1]

2. **base_ang_vel** (3 values)
   - Angular velocity in world frame: [wx, wy, wz]
   - Units: **rad/s**
   - Frame: World frame
   - Noise during training: Uniform [-0.2, 0.2]

3. **projected_gravity** (3 values)
   - Gravity vector projected into robot's body frame: [gx, gy, gz]
   - Values: Normalized gravity direction (unit vector)
   - Frame: Body frame (robot's local coordinate system)
   - Purpose: Tells robot its orientation relative to gravity
   - Noise during training: Uniform [-0.05, 0.05]
   - **CRITICAL**: Cannot be zeroed - robot needs this to balance!

4. **velocity_commands** (3 values)
   - Desired velocity command: [lin_vel_x, lin_vel_y, ang_vel_z]
   - Range: lin_vel_x/y: [-0.6, 0.6] m/s, ang_vel_z: [-0.8, 0.8] rad/s
   - Purpose: User input - where the robot should move
   - When idle: [0.0, 0.0, 0.0]

5. **joint_pos** (12 values)
   - Joint positions **relative to default positions**: [FL_hip, FL_upper, FL_lower, FR_hip, FR_upper, FR_lower, BL_hip, BL_upper, BL_lower, BR_hip, BR_upper, BR_lower]
   - Units: **Radians** (relative to default, not absolute)
   - Default positions (absolute):
     - FL: [0.1465, -0.1465, 0.0]
     - FR: [-0.1465, 0.1465, 0.0]
     - BL: [-0.1465, 0.1465, 0.0]
     - BR: [0.1465, -0.1465, 0.0]
   - Noise during training: Uniform [-0.01, 0.01]

6. **joint_vel** (12 values)
   - Joint velocities: [FL_hip_vel, FL_upper_vel, FL_lower_vel, ...] (same order as joint_pos)
   - Units: **Radians/second**
   - Noise during training: Uniform [-1.5, 1.5]

7. **actions** (12 values)
   - Last action taken by the policy: [a0, a1, ..., a11]
   - Range: [-1, 1] (raw model output before scaling)
   - Purpose: Provides temporal information to the policy
   - When idle/first step: Use zeros [0, 0, ..., 0] or previous action

**Total: 3 + 3 + 3 + 3 + 12 + 12 + 12 = 48 values**

---

## Key Differences

### 1. **Temporal History**
- **Old Model**: Uses 5 timesteps of historical data (60 joint angles + 30 orientation values)
- **Isaac Lab Model**: Uses only current timestep + last action (12 values)
- **Impact**: Isaac Lab model is more memory-efficient but relies on the policy network to learn temporal patterns

### 2. **Orientation Representation**
- **Old Model**: 
  - Uses 6 values: [shift, move, translate, yaw, roll, pitch]
  - From accelerometer: `shift = acc_x/16384.0`, `move = -acc_y/16384.0`, `translate = acc_z/16384.0`
  - From gyroscope: `yaw = -gyro_x/131.0`, `roll = gyro_y/131.0`, `pitch = -gyro_z/131.0`
  - Normalized: Accelerometer values divided by 30.0, gyroscope values divided by 500.0
  - Historical: 5 timesteps stored

- **Isaac Lab Model**:
  - Uses 6 values split differently:
    - `projected_gravity` (3): Normalized gravity vector in body frame [gx, gy, gz]
    - `base_ang_vel` (3): Angular velocity [wx, wy, wz] in rad/s
  - No temporal history - only current values
  - **Critical**: `projected_gravity` must be computed from accelerometer data

### 3. **Linear Velocity**
- **Old Model**: Not explicitly included
- **Isaac Lab Model**: Requires `base_lin_vel` [vx, vy, vz] in m/s
- **Note**: Can be zeroed if you don't have odometry, but performance will be degraded

### 4. **Joint Positions**
- **Old Model**: Uses historical normalized model outputs (actions) in range [-1, 1]
- **Isaac Lab Model**: Uses actual joint positions in **radians**, **relative to default positions**
- **Important**: You need encoders to read actual joint positions!

### 5. **Joint Velocities**
- **Old Model**: Not explicitly included
- **Isaac Lab Model**: Requires `joint_vel` in rad/s
- **Source**: Encoders with velocity feedback or differentiate position over time

### 6. **Commands**
- **Old Model**: 6D one-hot encoding [w, s, a, d, arrowleft, arrowright]
- **Isaac Lab Model**: 3D continuous values [lin_vel_x, lin_vel_y, ang_vel_z]
- **Conversion needed**: Map your discrete commands to continuous velocity commands

### 7. **Intensity**
- **Old Model**: Includes intensity as a single normalized value
- **Isaac Lab Model**: No intensity field - speed is controlled via `velocity_commands`

---

## Mapping Your Accelerometer Data to Isaac Lab Model

### From `accelerometer.py`:
```python
shift = acc_x/16384.0      # acc_x in raw sensor units
move = -acc_y/16384.0      # acc_y in raw sensor units (flipped)
translate = acc_z/16384.0  # acc_z in raw sensor units
yaw = -gyro_x/131.0        # gyro_x in raw sensor units (flipped)
roll = gyro_y/131.0        # gyro_y in raw sensor units
pitch = -gyro_z/131.0      # gyro_z in raw sensor units (flipped)
```

### To Isaac Lab Model:

#### 1. **projected_gravity** (3 values) - CRITICAL
```python
# From accelerometer readings (shift, move, translate)
# These are already in g units (divided by 16384.0)
accel_body = np.array([shift, move, translate])  # In g units

# Normalize to get gravity direction (unit vector)
projected_gravity = accel_body / np.linalg.norm(accel_body)

# If robot is stationary, this should point roughly [0, 0, -1] when upright
# (may need sign adjustments based on your coordinate system)
```

**Note**: The accelerometer gives you acceleration in body frame. When stationary, this is primarily gravity, so normalizing gives you the gravity direction.

#### 2. **base_ang_vel** (3 values)
```python
# From gyroscope readings (yaw, roll, pitch)
# These are in degrees/second (divided by 131.0)
gyro_deg_per_sec = np.array([yaw, roll, pitch])

# Convert to radians/second
base_ang_vel = np.deg2rad(gyro_deg_per_sec)
# Result: [wx, wy, wz] in rad/s
```

**Conversion**: 
- Your gyroscope: degrees/second (÷131.0 gives deg/s)
- Isaac Lab needs: radians/second
- Conversion: `rad/s = deg/s × (π/180)`

#### 3. **base_lin_vel** (3 values) - OPTIONAL
```python
# If you don't have odometry, you can zero this:
base_lin_vel = np.array([0.0, 0.0, 0.0])

# Or integrate accelerometer (with drift correction):
# This is more complex and requires filtering/integration
# For now, zeroing is acceptable but performance will be degraded
```

---

## Unit Conversions Summary

| Data | Old Model | Isaac Lab Model | Conversion |
|------|-----------|-----------------|------------|
| **Accelerometer** | Normalized [-1, 1] (÷30.0) | Normalized unit vector | Normalize `[shift, move, translate]` to unit vector |
| **Gyroscope** | Normalized [-1, 1] (÷500.0) | rad/s | `deg/s × (π/180)` |
| **Joint Positions** | Normalized actions [-1, 1] | Radians (relative) | Read from encoders, subtract defaults |
| **Joint Velocities** | Not used | rad/s | Read from encoders or differentiate |
| **Commands** | One-hot [6] | Continuous [3] | Map discrete to velocity |

---

## Command Mapping: Old → Isaac Lab

### Old Model Commands:
```python
[w, s, a, d, arrowleft, arrowright]  # One-hot encoding
```

### Isaac Lab Commands:
```python
[lin_vel_x, lin_vel_y, ang_vel_z]  # Continuous values
```

### Suggested Mapping:
```python
def map_commands_to_velocity(old_commands):
    """
    Convert old one-hot commands to Isaac Lab velocity commands.
    
    Args:
        old_commands: [w, s, a, d, arrowleft, arrowright] (0.0 or 1.0)
    
    Returns:
        [lin_vel_x, lin_vel_y, ang_vel_z]
    """
    lin_vel_x = 0.0
    lin_vel_y = 0.0
    ang_vel_z = 0.0
    
    # Forward/backward
    if old_commands[0] > 0.5:  # 'w' - forward
        lin_vel_x = 0.4  # m/s (adjust based on your robot's speed)
    elif old_commands[1] > 0.5:  # 's' - backward
        lin_vel_x = -0.4  # m/s
    
    # Left/right strafe
    if old_commands[2] > 0.5:  # 'a' - left
        lin_vel_y = 0.3  # m/s (adjust based on your robot)
    elif old_commands[3] > 0.5:  # 'd' - right
        lin_vel_y = -0.3  # m/s
    
    # Rotation
    if old_commands[4] > 0.5:  # 'arrowleft' - turn left
        ang_vel_z = 0.5  # rad/s (adjust based on your robot)
    elif old_commands[5] > 0.5:  # 'arrowright' - turn right
        ang_vel_z = -0.5  # rad/s
    
    # Clamp to Isaac Lab ranges
    lin_vel_x = np.clip(lin_vel_x, -0.6, 0.6)
    lin_vel_y = np.clip(lin_vel_y, -0.6, 0.6)
    ang_vel_z = np.clip(ang_vel_z, -0.8, 0.8)
    
    return np.array([lin_vel_x, lin_vel_y, ang_vel_z])
```

---

## What You Need to Add/Change

### ✅ **Already Have (from accelerometer.py):**
1. ✅ Accelerometer data (shift, move, translate)
2. ✅ Gyroscope data (yaw, roll, pitch)
3. ✅ Can compute `projected_gravity` and `base_ang_vel`

### ⚠️ **Need to Add/Verify:**
1. ✅ **Joint position reading** - Using last commanded position from `config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE']`
   - **Decision**: Servos don't support position feedback, so using commanded positions is acceptable
   - **Note**: This is an approximation - actual positions may lag behind commands, but it's better than nothing
   - **Implementation**: Use the helper function `get_joint_positions_from_config()` provided below
2. ⚠️ **Joint velocity** - Differentiate commanded positions over time (see helper function below)
3. ⚠️ **base_lin_vel** - Can zero for now, but better to integrate accelerometer

### 🔄 **Need to Change:**
1. 🔄 Remove temporal history (no more 5-timestep storage)
2. 🔄 Convert commands from one-hot to continuous velocity
3. 🔄 Change normalization: accelerometer → unit vector, gyroscope → rad/s
4. 🔄 Use actual joint positions instead of historical actions

---

## Helper Function: Getting Joint Positions from Your Config

```python
import numpy as np
import utilities.config as config

def get_joint_positions_from_config():
    """
    Get current joint positions from your SERVO_CONFIG.
    Returns joint positions in radians in the order: [FL_hip, FL_upper, FL_lower, FR_hip, FR_upper, FR_lower, BL_hip, BL_upper, BL_lower, BR_hip, BR_upper, BR_lower]
    """
    joint_positions = []
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        for joint_name in ['hip', 'upper', 'lower']:
            angle = config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE']
            joint_positions.append(angle)
    return np.array(joint_positions, dtype=np.float32)

def compute_joint_velocities(current_positions, previous_positions, dt=0.033):
    """
    Compute joint velocities by differentiating positions.
    
    Args:
        current_positions: [12] current joint positions in radians
        previous_positions: [12] previous joint positions in radians
        dt: time step in seconds (default 0.033 for ~30 Hz)
    
    Returns:
        [12] joint velocities in rad/s
    """
    return (current_positions - previous_positions) / dt
```

---

## Example: Constructing Isaac Lab Observation

```python
import numpy as np

def construct_isaac_lab_observation(
    # From accelerometer.py
    shift, move, translate, yaw, roll, pitch,
    # From joint positions (using CURRENT_ANGLE from config - commanded positions)
    # Note: You'll need to call get_joint_positions_from_config() to get these
    previous_joint_pos,  # [12] previous joint positions for velocity calculation
    # From commands
    old_commands,         # [6] one-hot commands
    # From previous inference
    last_action           # [12] previous model output [-1, 1]
):
    """
    Construct 48-dim observation vector for Isaac Lab model.
    
    Note: Uses commanded joint positions (not actual positions) since servos
    don't support position feedback.
    """
    
    # Get current joint positions from config (commanded positions)
    joint_pos_absolute = get_joint_positions_from_config()
    
    # 1. base_lin_vel (3) - Zero for now (can improve later)
    base_lin_vel = np.array([0.0, 0.0, 0.0])
    
    # 2. base_ang_vel (3) - From gyroscope
    gyro_deg_per_sec = np.array([yaw, roll, pitch])
    base_ang_vel = np.deg2rad(gyro_deg_per_sec)  # Convert to rad/s
    
    # 3. projected_gravity (3) - From accelerometer
    accel_body = np.array([shift, move, translate])  # In g units
    norm = np.linalg.norm(accel_body)
    if norm > 0.01:  # Avoid division by zero
        projected_gravity = accel_body / norm
    else:
        projected_gravity = np.array([0.0, 0.0, -1.0])  # Default: upright
    
    # 4. velocity_commands (3) - From old commands
    velocity_commands = map_commands_to_velocity(old_commands)
    
    # 5. joint_pos (12) - Relative to defaults
    default_positions = np.array([
        0.1465, -0.1465, 0.0,    # FL
        -0.1465, 0.1465, 0.0,    # FR
        -0.1465, 0.1465, 0.0,    # BL
        0.1465, -0.1465, 0.0     # BR
    ])
    joint_pos = joint_pos_absolute - default_positions
    
    # 6. joint_vel (12) - Compute from position difference
    dt = 0.033  # ~30 Hz update rate (adjust to match your loop rate)
    joint_vel = compute_joint_velocities(joint_pos_absolute, previous_joint_pos, dt)
    
    # 7. actions (12) - Previous model output
    # (last_action parameter)
    
    # Concatenate all
    obs = np.concatenate([
        base_lin_vel,      # 3
        base_ang_vel,      # 3
        projected_gravity, # 3
        velocity_commands, # 3
        joint_pos,         # 12
        joint_vel,         # 12
        last_action        # 12
    ])
    
    assert len(obs) == 48, f"Observation must be 48 dims, got {len(obs)}"
    return obs, joint_pos_absolute  # Return current positions for next iteration
```

**Note**: You'll need to store `joint_pos_absolute` between calls to compute velocities. Initialize it at startup with current positions or zeros.
```

---

## Summary Checklist

- [x] **Understand the differences**: Old model (97 dims) vs Isaac Lab (48 dims)
- [ ] **Map accelerometer data**: Convert to `projected_gravity` (unit vector) and `base_ang_vel` (rad/s)
- [x] **Get joint positions**: Using commanded positions from `config.SERVO_CONFIG` (decided)
- [ ] **Compute joint velocities**: Differentiate commanded positions over time
- [ ] **Convert commands**: One-hot → continuous velocity [lin_vel_x, lin_vel_y, ang_vel_z]
- [ ] **Remove temporal history**: No more 5-timestep storage needed
- [ ] **Handle base_lin_vel**: Zero for now, or integrate accelerometer later
- [ ] **Test observation construction**: Ensure 48-dim vector is correct

---

## Notes

1. **No Noise During Inference**: Isaac Lab adds noise during training but NOT during inference. Use clean sensor readings.

2. **Coordinate System**: Verify your accelerometer/gyroscope coordinate system matches Isaac Lab's expectations. You may need to swap axes or flip signs.

3. **Joint Positions**: Isaac Lab uses positions **relative to defaults**, not absolute. Make sure to subtract the default positions.

4. **Actions**: Store the last model output (12 values in [-1, 1]) to use as the `actions` field in the next observation.

5. **When Idle**: Set `velocity_commands = [0.0, 0.0, 0.0]` and `actions` can be zeros or the last action.

---

## Understanding Joint Encoders vs Your Current Setup

### What are "Joint Encoders"?
**Joint encoders** are sensors that measure the **actual current angle/position** of each joint on your robot. They tell you where the joint **really is**, not where you told it to go.

### Your Current Setup
Looking at your code, you're currently:
1. **Commanding** servos to move to target angles via `set_target()` in `servos.py`
2. **Storing** the commanded angle in `config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE']`

**Important**: `CURRENT_ANGLE` is what you **commanded**, not necessarily what the servo **actually is**. If a servo is slow, blocked, or has backlash, the actual position might differ from the commanded position.

### Options for Getting Actual Joint Positions

#### ✅ **Your Approach: Use Commanded Positions**
Since your servos don't support position feedback, you'll use the last commanded position stored in `config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE']`.

**Pros:**
- ✅ No hardware changes needed
- ✅ Already implemented in your code
- ✅ Works well if servos are fast and accurate

**Cons:**
- ⚠️ Actual position may lag behind commanded position (especially during fast movements)
- ⚠️ Won't detect if a servo is blocked or fails to reach target
- ⚠️ May have some error, but acceptable for getting started

**Implementation:** Use the helper function `get_joint_positions_from_config()` provided below.

### For Joint Velocities
Once you have positions, you can compute velocities by differentiating:
```python
# Store previous positions
prev_joint_pos = joint_positions.copy()

# Later, compute velocity
dt = 0.033  # ~30 Hz update rate
joint_velocities = (joint_positions - prev_joint_pos) / dt
```

---

## Important Notes About Using Commanded Positions

### Limitations to Be Aware Of

1. **Position Lag**: During fast movements, actual servo positions may lag behind commanded positions. The model might think joints are in different positions than they actually are.

2. **No Error Detection**: If a servo fails to move (blocked, broken, etc.), you won't know - the model will think it's at the commanded position.

3. **Backlash**: Mechanical backlash in gears/joints means actual position might differ slightly from commanded, especially when changing direction.

### Mitigation Strategies

- **Keep movements smooth**: Avoid sudden large movements that cause significant lag
- **Monitor robot behavior**: If the robot behaves unexpectedly, position lag might be the cause
- **Consider slower control loop**: If servos are slow, you might need to slow down your control frequency to match servo speed

### Future Improvements (Optional)

If you find that position lag is causing issues, you could:
- Add rotary encoders later for true position feedback
- Use a Kalman filter to estimate actual positions based on commanded positions and time
- Add IMU-based pose estimation to help compensate

---

## Questions to Consider

1. ✅ **Joint positions**: Using commanded positions from config (decided)
2. **What coordinate system does your IMU use?** Verify it matches Isaac Lab's expectations
3. **How fast does your robot move?** Adjust the velocity command mapping accordingly
4. **Do you want to integrate accelerometer for `base_lin_vel`?** Or zero it for now?
5. **What's your control loop frequency?** Make sure it matches your servo response time

---

**Good luck with your migration!** 🚀
