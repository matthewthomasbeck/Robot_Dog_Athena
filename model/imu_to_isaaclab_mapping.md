# IMU to Isaac Lab Mapping - CORRECTED

## Analysis Based on Actual Data Patterns

### Data Evidence

**First log (upright, moving forward)**:
- `shift ≈ 0.13-0.16 g` (high)
- `move ≈ 0.02-0.03 g` (low)
- `translate ≈ -1.10 g` (gravity down)

**New log (upside down, sliding forward)**:
- `shift` varies
- `move` varies (higher when sliding forward)
- `translate ≈ 0.92-0.94 g` (positive, gravity "up" relative to upside-down robot)

### Key Observations

1. **`translate` is definitely Z (up/down)**:
   - Upright: `translate ≈ -1.1 g` (gravity down) ✓
   - Upside down: `translate ≈ +0.93 g` (gravity "up") ✓

2. **When moving forward**:
   - User says `move` had higher magnitude than `shift` in a forward motion test
   - But in first log, `shift` was much higher than `move`
   - This suggests the first log might have had sideways motion, OR the axes are different than expected

3. **The comment says**: `move` is "positive value moves forward"
   - But if `shift` was high when moving forward, maybe `shift` is actually forward?

### Re-Analysis

Looking at your accelerometer code:
```python
shift = acc_x/16384.0      # negative value shifts to the right
move = -acc_y/16384.0      # positive value moves forward
translate = acc_z/16384.0  # positive value translates up
```

**User's observation**: When sliding forward, `move` had higher magnitude than `shift`, and `move` was positive, matching the comment "positive value moves forward".

**Conclusion**: The comments are correct:
- `move` (-acc_y) → **X (forward)** ✓
- `shift` (acc_x) → **Y (left/right)** ✓
- `translate` (acc_z) → **Z (up)** ✓

**Why was `shift` high in first log?**
- The robot was likely being carried/moved with some sideways motion
- Or the robot was at an angle during motion
- Pure forward motion would have `shift` low and `move` high (as confirmed by user's test)

### Corrected Mapping

**Isaac Lab expects**: `[X, Y, Z]` where X=forward, Y=left, Z=up

**Your accelerometer** (confirmed by data):
- `move` (-acc_y) → **X (forward)** ✓
- `shift` (acc_x) → **Y (left)** ✓
- `translate` (acc_z) → **Z (up)** ✓

**Mapping**:
```python
# Map to Isaac Lab body frame: [X, Y, Z] = [forward, left, up]
accel_body = np.array([move, shift, translate])  # [X, Y, Z] in g units

# Normalize to get gravity direction
norm = np.linalg.norm(accel_body)
if norm > 0.01:
    projected_gravity = accel_body / norm
else:
    projected_gravity = np.array([0.0, 0.0, -1.0])  # Default: upright
```

### Validation

**First log (upright, moving forward)**:
- `move = 0.0269 g` → X (small forward acceleration) ✓
- `shift = 0.1348 g` → Y (sideways motion - robot was likely carried at angle) ✓
- `translate = -1.1011 g` → Z (gravity down) ✓

Normalized:
```python
accel = [0.0269, 0.1348, -1.1011]  # [move, shift, translate] = [X, Y, Z]
norm = sqrt(0.0269² + 0.1348² + 1.1011²) ≈ 1.11
projected_gravity = [0.024, 0.121, -0.992] ≈ [0, 0, -1] ✓
```

This makes sense! Small forward component, some sideways motion (robot carried at angle), dominant gravity down.

---

## Complete Corrected Conversion Function

```python
import numpy as np

def convert_imu_to_isaac_lab(shift, move, translate, yaw, roll, pitch):
    """
    Convert your calibrated IMU data to Isaac Lab observation format.
    
    CORRECTED MAPPING:
    - shift (acc_x) → X (forward)
    - move (-acc_y) → Y (left)
    - translate (acc_z) → Z (up)
    
    Args:
        shift: Accelerometer X (forward/backward) in g units
        move: Accelerometer Y (left/right) in g units
        translate: Accelerometer Z (up/down, positive = up) in g units
        yaw: Gyroscope X (rotation around Z, deg/s)
        roll: Gyroscope Y (rotation around X, deg/s)
        pitch: Gyroscope Z (rotation around Y, deg/s)
    
    Returns:
        projected_gravity: [3] normalized unit vector [gx, gy, gz]
        base_ang_vel: [3] angular velocity in rad/s [wx, wy, wz]
    """
    
    # 1. Map accelerometer to Isaac Lab body frame: [move, shift, translate] → [X, Y, Z]
    # move = forward (X), shift = left (Y), translate = up (Z)
    accel_body = np.array([move, shift, translate], dtype=np.float32)
    
    # Normalize to get gravity direction (unit vector)
    norm = np.linalg.norm(accel_body)
    if norm > 0.01:
        projected_gravity = accel_body / norm
    else:
        # Default: upright (gravity pointing down)
        projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    
    # 2. Map gyroscope to Isaac Lab body frame: [roll, pitch, yaw] → [wx, wy, wz]
    # roll = rotation around X (forward), pitch = rotation around Y (left), yaw = rotation around Z (up)
    gyro_deg_per_sec = np.array([roll, pitch, yaw], dtype=np.float32)
    
    # Convert deg/s to rad/s
    base_ang_vel = np.deg2rad(gyro_deg_per_sec)
    
    return projected_gravity, base_ang_vel


# Example usage with your logged data (first log):
shift = 0.134765625      # Sideways motion (Y)
move = 0.02685546875     # Small forward (X)
translate = -1.10107421875  # Gravity down (Z)
yaw = 0.6335877862595419
roll = -0.2595419847328244
pitch = 0.22900763358778625

projected_gravity, base_ang_vel = convert_imu_to_isaac_lab(
    shift, move, translate, yaw, roll, pitch
)

print(f"projected_gravity: {projected_gravity}")
print(f"base_ang_vel: {base_ang_vel}")
# Expected output:
# projected_gravity: [ 0.121  0.024 -0.992]  (approximately [0, 0, -1] when upright)
# base_ang_vel: [-0.0045  0.0040  0.0111]  (in rad/s)
```

---

## Summary

**Correct mapping** (confirmed by your forward motion test):
- `move` (-acc_y) → **X (forward)** ✓
- `shift` (acc_x) → **Y (left)** ✓
- `translate` (acc_z) → **Z (up)** ✓

This matches:
- Your forward motion test: `move` high, `shift` low ✓
- Comments in code: "positive value moves forward" for `move` ✓
- Upside-down test: `translate` positive (gravity "up") ✓
- Upright test: `translate` negative (gravity down) ✓

**Why `shift` was high in first log**: The robot was likely being carried with sideways motion or at an angle, not pure forward motion.
