# IMU Data Analysis: Real Robot vs Isaac Lab Expectations

## Overview

This document analyzes the logged IMU data from your physical robot and compares it to what Isaac Lab's observation space expects, with special attention to coordinate system alignment.

---

## Logged IMU Data Format

From `robot_dog.log`, the IMU data format is:
```
[shift, move, translate, yaw, roll, pitch]
```

### Example Data (Robot Moving Forward, Upright)
```
[0.134765625, 0.02685546875, -1.10107421875, 0.6335877862595419, -0.2595419847328244, 0.22900763358778625]
```

### Data Source (from `accelerometer.py`)
```python
shift = acc_x/16384.0      # acc_x in raw sensor units → g units
move = -acc_y/16384.0      # acc_y in raw sensor units (flipped) → g units
translate = acc_z/16384.0  # acc_z in raw sensor units → g units

yaw = -gyro_x/131.0        # gyro_x in raw sensor units (flipped) → deg/s
roll = gyro_y/131.0        # gyro_y in raw sensor units → deg/s
pitch = -gyro_z/131.0      # gyro_z in raw sensor units (flipped) → deg/s
```

### Interpretation of Logged Data

When you were carrying the robot forward and upright:

1. **Accelerometer (shift, move, translate)**:
   - `shift ≈ 0.13-0.16 g`: Small positive acceleration in +X direction (forward acceleration)
   - `move ≈ 0.02-0.03 g`: Very small sideways acceleration (mostly forward motion)
   - `translate ≈ -1.10 g`: **Gravity pointing down** (dominant component)

2. **Gyroscope (yaw, roll, pitch)**:
   - `yaw ≈ 0.6-4.8 deg/s`: Rotation around Z-axis (turning)
   - `roll ≈ -0.3 to 6.5 deg/s`: Rotation around X-axis (sideways tilt)
   - `pitch ≈ -4.6 to 2.3 deg/s`: Rotation around Y-axis (forward/backward tilt)

**Key Observation**: When stationary and upright, we'd expect:
- `translate ≈ -1.0 g` (gravity down)
- `shift ≈ 0` (no forward acceleration)
- `move ≈ 0` (no sideways acceleration)

The fact that `translate ≈ -1.1 g` and `shift ≈ 0.13-0.16 g` suggests:
- Robot is mostly upright (gravity dominates in Z)
- Small forward acceleration from being carried

---

## Isaac Lab IMU Expectations

Isaac Lab expects two IMU-derived observations:

### 1. `projected_gravity` (3 values)
- **Type**: Normalized unit vector
- **Frame**: Body frame (robot's local coordinate system)
- **Units**: Unitless (normalized)
- **Expected when upright**: Approximately `[0, 0, -1]` or `[0, 0, 1]` depending on coordinate system
- **Computation in Isaac Lab**: 
  ```python
  # Gravity in world frame: [0, 0, -1] (pointing down)
  gravity_w = [0, 0, -1]
  # Rotate to body frame using inverse quaternion
  projected_gravity_b = quat_apply_inverse(robot_quat_w, gravity_w)
  ```

### 2. `base_ang_vel` (3 values)
- **Type**: Angular velocity vector
- **Frame**: Body frame
- **Units**: **radians/second**
- **Order**: `[wx, wy, wz]` (angular velocity around X, Y, Z axes)

---

## Coordinate System Analysis

### Isaac Lab World Frame
- **+X**: Forward
- **+Y**: Left
- **+Z**: Up
- **Gravity**: `[0, 0, -1]` in world frame (pointing down)

### Your Robot's Coordinate System

**Important**: You mentioned that Isaac Lab spawns the robot turned 90° from where it thinks is the front, so "the code thinks the left side of the robot is the face."

This suggests:
- **Isaac Lab expects**: +X forward
- **Your robot's actual front**: +Y direction (left side in Isaac Lab's frame)
- **Rotation**: 90° around Z-axis

### Your Accelerometer Coordinate System

From your code comments in `accelerometer.py`:
- `shift = acc_x/16384.0`: "negative value shifts to the right"
- `move = -acc_y/16384.0`: "positive value moves forward" (flipped)
- `translate = acc_z/16384.0`: "positive value translates up"

This suggests your accelerometer's coordinate system (before the sign flips):
- **acc_x**: Right/left (negative = right)
- **acc_y**: Forward/backward (negative = forward, but flipped so positive = forward)
- **acc_z**: Up/down (positive = up)

After your sign flips:
- **shift (acc_x)**: Right/left (negative = right)
- **move (-acc_y)**: Forward/backward (positive = forward)
- **translate (acc_z)**: Up/down (positive = up)

### Mapping to Isaac Lab Body Frame

Isaac Lab expects the body frame to match the world frame orientation when the robot is at its default pose. However, if your robot is rotated 90° in simulation, we need to account for this.

**Hypothesis**: Your accelerometer's coordinate system might be:
- **shift (acc_x)**: Maps to Isaac Lab's **+Y** (left)
- **move (-acc_y)**: Maps to Isaac Lab's **+X** (forward)
- **translate (acc_z)**: Maps to Isaac Lab's **+Z** (up)

But we need to verify this based on your calibration.

---

## Data Comparison: Real vs Expected

### When Robot is Stationary and Upright

**Your IMU (from log, when mostly stationary)**:
```
shift ≈ 0.13-0.16 g      # Small forward acceleration
move ≈ 0.02-0.03 g       # Very small sideways
translate ≈ -1.10 g      # Gravity down
```

**Isaac Lab Expected (when upright)**:
```
projected_gravity ≈ [0, 0, -1]  # Normalized unit vector pointing down
```

**Analysis**:
- Your `translate ≈ -1.10 g` is close to `-1.0 g` (gravity), which is good!
- The small `shift` and `move` values are likely from motion/noise
- To get `projected_gravity`, we need to normalize: `[shift, move, translate] / ||[shift, move, translate]||`

### When Robot is Moving Forward

**Your IMU (from log)**:
```
shift ≈ 0.13-0.16 g      # Forward acceleration
move ≈ 0.02-0.03 g       # Small sideways
translate ≈ -1.10 g      # Gravity (still dominant)
```

**For `projected_gravity`**:
- The accelerometer reading includes both gravity and linear acceleration
- When moving at constant velocity, linear acceleration is small, so gravity dominates
- Normalizing gives us the gravity direction: `projected_gravity ≈ normalize([0.13, 0.03, -1.10])`

**For `base_ang_vel`**:
- Your gyroscope: `[yaw, roll, pitch]` in **deg/s**
- Isaac Lab needs: `[wx, wy, wz]` in **rad/s**
- Conversion: `rad/s = deg/s × (π/180)`

---

## Critical Coordinate System Questions

### 1. What is your accelerometer's coordinate system?

Based on your code, it seems:
- **shift (acc_x)**: Right/left axis
- **move (-acc_y)**: Forward/backward axis  
- **translate (acc_z)**: Up/down axis

But we need to verify:
- Does `shift` positive mean right or left?
- Does `move` positive mean forward or backward?
- Does `translate` positive mean up or down?

### 2. How does your accelerometer coordinate system map to Isaac Lab's body frame?

Isaac Lab expects:
- **+X**: Forward
- **+Y**: Left
- **+Z**: Up

If your robot is rotated 90° in simulation (left side is front), then:
- Your robot's actual front might be Isaac Lab's +Y
- Your robot's actual right might be Isaac Lab's -X

### 3. What about the gyroscope?

Your gyroscope gives:
- **yaw**: Rotation around Z-axis (deg/s)
- **roll**: Rotation around X-axis (deg/s)
- **pitch**: Rotation around Y-axis (deg/s)

Isaac Lab expects:
- **wx**: Angular velocity around X-axis (rad/s)
- **wy**: Angular velocity around Y-axis (rad/s)
- **wz**: Angular velocity around Z-axis (rad/s)

**Mapping might be**:
- `yaw` → `wz` (both around Z-axis)
- `roll` → `wx` (both around X-axis)
- `pitch` → `wy` (both around Y-axis)

But we need to verify the sign conventions match.

---

## Recommended Mapping Strategy

### Step 1: Determine Coordinate System Mapping

Since you've calibrated the accelerometer and know its orientation, you should:

1. **Test with robot stationary and upright**:
   - Read accelerometer: `[shift, move, translate]`
   - Expected: Should be approximately `[0, 0, -1]` in g units (gravity down)
   - This tells us which axis is "up" and the sign convention

2. **Test with robot tilted forward**:
   - Read accelerometer
   - Expected: `translate` should decrease, `move` (or `shift`) should change
   - This tells us which axis is "forward"

3. **Test with robot rotated**:
   - Read gyroscope during rotation
   - Expected: One of `[yaw, roll, pitch]` should be large
   - This tells us the gyroscope axis mapping

### Step 2: Create Mapping Function

Based on your calibration, create a function that maps your IMU data to Isaac Lab format:

```python
def map_imu_to_isaac_lab(shift, move, translate, yaw, roll, pitch):
    """
    Map your IMU data to Isaac Lab format.
    
    Args:
        shift, move, translate: Accelerometer in g units
        yaw, roll, pitch: Gyroscope in deg/s
    
    Returns:
        projected_gravity: [3] normalized unit vector
        base_ang_vel: [3] angular velocity in rad/s
    """
    
    # 1. Map accelerometer to Isaac Lab body frame
    # TODO: Adjust based on your coordinate system
    # Hypothesis: [shift, move, translate] → [Y, X, Z] or [X, Y, Z]?
    accel_body = np.array([move, shift, translate])  # Example: might need adjustment
    
    # Normalize to get gravity direction
    norm = np.linalg.norm(accel_body)
    if norm > 0.01:
        projected_gravity = accel_body / norm
    else:
        projected_gravity = np.array([0.0, 0.0, -1.0])  # Default: upright
    
    # 2. Map gyroscope to Isaac Lab body frame
    # TODO: Adjust based on your coordinate system
    # Hypothesis: [yaw, roll, pitch] → [wz, wx, wy]?
    gyro_deg_per_sec = np.array([roll, pitch, yaw])  # Example: might need adjustment
    
    # Convert to rad/s
    base_ang_vel = np.deg2rad(gyro_deg_per_sec)
    
    return projected_gravity, base_ang_vel
```

### Step 3: Verify with Test Cases

Test the mapping with known orientations:

1. **Upright, stationary**:
   - Expected `projected_gravity ≈ [0, 0, -1]` or `[0, 0, 1]`
   - Expected `base_ang_vel ≈ [0, 0, 0]`

2. **Tilted forward 45°**:
   - Expected `projected_gravity` should have forward component
   - Expected `base_ang_vel` should be small (if stationary)

3. **Rotating around vertical axis**:
   - Expected `base_ang_vel[2]` (wz) should be large
   - Expected `projected_gravity` should remain roughly `[0, 0, -1]`

---

## Key Differences Summary

| Aspect | Your IMU | Isaac Lab |
|--------|----------|-----------|
| **Accelerometer Units** | g units (÷16384.0) | Normalized unit vector |
| **Accelerometer Format** | `[shift, move, translate]` | `[gx, gy, gz]` (normalized) |
| **Gyroscope Units** | deg/s (÷131.0) | rad/s |
| **Gyroscope Format** | `[yaw, roll, pitch]` | `[wx, wy, wz]` |
| **Coordinate System** | Your robot's body frame | Isaac Lab's body frame (may be rotated) |
| **Temporal History** | 5 timesteps stored | Only current timestep |

---

## Next Steps

1. **Verify coordinate system mapping**:
   - Test accelerometer with known orientations
   - Test gyroscope with known rotations
   - Document the mapping between your IMU and Isaac Lab's expected frame

2. **Implement mapping function**:
   - Create function to convert your IMU data to Isaac Lab format
   - Account for 90° rotation if needed
   - Test with your logged data

3. **Validate with simulation**:
   - Compare your real IMU data to Isaac Lab simulation IMU data
   - Ensure `projected_gravity` and `base_ang_vel` match in behavior

4. **Handle coordinate system rotation**:
   - If your robot is rotated 90° in simulation, apply rotation to IMU data
   - Or adjust Isaac Lab's robot spawn orientation

---

## Important Notes

1. **Accelerometer includes both gravity and linear acceleration**:
   - When moving, the accelerometer reading = gravity + linear_acceleration
   - For `projected_gravity`, we normalize the reading, which works if linear acceleration is small compared to gravity
   - During fast acceleration, this approximation may degrade

2. **Coordinate system rotation**:
   - The 90° rotation you mentioned is critical
   - You may need to rotate the IMU data to match Isaac Lab's expected frame
   - Or adjust the robot's spawn orientation in simulation

3. **Sign conventions**:
   - Verify all sign conventions match between your IMU and Isaac Lab
   - Test with known orientations to confirm

4. **Noise and filtering**:
   - Your logged data shows some noise (small variations)
   - Isaac Lab adds noise during training but not during inference
   - Consider light filtering if noise is significant

---

## Questions to Answer

1. **What is the exact coordinate system of your accelerometer?**
   - Which axis is forward? (move or shift?)
   - Which axis is up? (translate?)
   - What are the sign conventions?

2. **How does your accelerometer coordinate system map to Isaac Lab's body frame?**
   - Does `[shift, move, translate]` map to `[X, Y, Z]` or `[Y, X, Z]`?
   - Do we need to apply a rotation?

3. **How does your gyroscope coordinate system map to Isaac Lab's body frame?**
   - Does `[yaw, roll, pitch]` map to `[wz, wx, wy]`?
   - Do the signs match?

4. **What is the effect of the 90° rotation?**
   - Should we rotate the IMU data before passing to the model?
   - Or should we adjust the robot's spawn orientation in simulation?

---

**Recommendation**: Start by testing your accelerometer with the robot in known orientations (upright, tilted forward, tilted sideways) to determine the exact coordinate system mapping. Then create a mapping function and validate it with your logged data.
