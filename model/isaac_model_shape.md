# Isaac Lab Model Shape - Complete Specification

## 48-Dimensional Observation Vector (Input to Model)

### Index 0-2: `base_lin_vel` (3 values) - Linear Velocity in World Frame
- **Units**: meters/second (m/s)
- **Frame**: World frame (global coordinate system)
- **Description**: Linear velocity of the robot's base/body center in world coordinates

**Index 0**: `base_lin_vel[0]` = `vx` (forward/backward velocity in world X)
- Positive: Moving forward in world X
- Negative: Moving backward in world X
- Current implementation: Always `0.0` (no odometry available)

**Index 1**: `base_lin_vel[1]` = `vy` (left/right velocity in world Y)
- Positive: Moving left in world Y
- Negative: Moving right in world Y
- Current implementation: Always `0.0` (no odometry available)

**Index 2**: `base_lin_vel[2]` = `vz` (up/down velocity in world Z)
- Positive: Moving up
- Negative: Moving down
- Current implementation: Always `0.0` (no odometry available)

---

### Index 3-5: `base_ang_vel` (3 values) - Angular Velocity in World Frame
- **Units**: radians/second (rad/s)
- **Frame**: World frame
- **Description**: Angular velocity (rotation rate) of the robot's base around world axes

**Index 3**: `base_ang_vel[0]` = `wx` (angular velocity around world X-axis)
- Positive: Rotating forward (pitch up)
- Negative: Rotating backward (pitch down)
- Source: Your gyroscope `roll` converted from deg/s to rad/s
- Conversion: `roll_deg_per_sec × (π/180) = wx`

**Index 4**: `base_ang_vel[1]` = `wy` (angular velocity around world Y-axis)
- Positive: Rotating left (roll left)
- Negative: Rotating right (roll right)
- Source: Your gyroscope `pitch` converted from deg/s to rad/s
- Conversion: `pitch_deg_per_sec × (π/180) = wy`

**Index 5**: `base_ang_vel[2]` = `wz` (angular velocity around world Z-axis)
- Positive: Rotating counterclockwise (yaw left)
- Negative: Rotating clockwise (yaw right)
- Source: Your gyroscope `yaw` converted from deg/s to rad/s
- Conversion: `yaw_deg_per_sec × (π/180) = wz`

---

### Index 6-8: `projected_gravity` (3 values) - Gravity Direction in Body Frame
- **Units**: Unitless (normalized unit vector)
- **Frame**: Body frame (robot's local coordinate system)
- **Description**: Direction of gravity vector projected into the robot's body frame
- **Range**: Each component in [-1, 1], vector magnitude = 1.0

**Index 6**: `projected_gravity[0]` = `gx` (gravity component along body X-axis)
- Positive: Gravity pulling forward (robot tilted backward)
- Negative: Gravity pulling backward (robot tilted forward)
- Source: Your accelerometer `move` (forward/backward axis)
- Calculation: `normalize([move, shift, translate])[0]`

**Index 7**: `projected_gravity[1]` = `gy` (gravity component along body Y-axis)
- Positive: Gravity pulling left (robot tilted right)
- Negative: Gravity pulling right (robot tilted left)
- Source: Your accelerometer `shift` (left/right axis)
- Calculation: `normalize([move, shift, translate])[1]`

**Index 8**: `projected_gravity[2]` = `gz` (gravity component along body Z-axis)
- Positive: Gravity pulling up (robot upside down)
- Negative: Gravity pulling down (robot upright)
- Source: Your accelerometer `translate` (up/down axis)
- Calculation: `normalize([move, shift, translate])[2]`
- When upright: Should be approximately `-1.0` (gravity down)

---

### Index 9-11: `velocity_commands` (3 values) - Desired Velocity Commands
- **Units**: m/s for linear, rad/s for angular
- **Frame**: Body frame
- **Description**: User's desired velocity command (where the robot should move)

**Index 9**: `velocity_commands[0]` = `lin_vel_x` (desired forward/backward velocity)
- Range: [-0.6, 0.6] m/s
- Positive: Move forward
- Negative: Move backward
- Source: Maps from discrete commands:
  - `'w'` → `+0.4` m/s
  - `'s'` → `-0.4` m/s
  - No command → `0.0`

**Index 10**: `velocity_commands[1]` = `lin_vel_y` (desired left/right velocity)
- Range: [-0.6, 0.6] m/s
- Positive: Move left
- Negative: Move right
- Source: Maps from discrete commands:
  - `'a'` → `+0.3` m/s
  - `'d'` → `-0.3` m/s
  - No command → `0.0`

**Index 11**: `velocity_commands[2]` = `ang_vel_z` (desired yaw rotation rate)
- Range: [-0.8, 0.8] rad/s
- Positive: Rotate counterclockwise (turn left)
- Negative: Rotate clockwise (turn right)
- Source: Maps from discrete commands:
  - `'arrowleft'` → `+0.5` rad/s
  - `'arrowright'` → `-0.5` rad/s
  - No command → `0.0`

---

### Index 12-23: `joint_pos` (12 values) - Joint Positions Relative to Defaults
- **Units**: Radians (relative to default positions, not absolute)
- **Frame**: Joint space
- **Description**: Current joint angles minus default/neutral joint angles
- **Order**: Depends on `JOINT_ORDERING_SCHEME` (currently `"by_type"`)

With `JOINT_ORDERING_SCHEME = "by_type"`:

**Index 12**: `joint_pos[0]` = `FL_hip` relative to default
- Absolute: `config.SERVO_CONFIG['FL']['hip']['CURRENT_ANGLE']`
- Default: `0.1465` radians
- Relative: `absolute - 0.1465`

**Index 13**: `joint_pos[1]` = `FR_hip` relative to default
- Absolute: `config.SERVO_CONFIG['FR']['hip']['CURRENT_ANGLE']`
- Default: `-0.1465` radians
- Relative: `absolute - (-0.1465)`

**Index 14**: `joint_pos[2]` = `BL_hip` relative to default
- Absolute: `config.SERVO_CONFIG['BL']['hip']['CURRENT_ANGLE']`
- Default: `-0.1465` radians
- Relative: `absolute - (-0.1465)`

**Index 15**: `joint_pos[3]` = `BR_hip` relative to default
- Absolute: `config.SERVO_CONFIG['BR']['hip']['CURRENT_ANGLE']`
- Default: `0.1465` radians
- Relative: `absolute - 0.1465`

**Index 16**: `joint_pos[4]` = `FL_upper` relative to default
- Absolute: `config.SERVO_CONFIG['FL']['upper']['CURRENT_ANGLE']`
- Default: `-0.1465` radians
- Relative: `absolute - (-0.1465)`

**Index 17**: `joint_pos[5]` = `FR_upper` relative to default
- Absolute: `config.SERVO_CONFIG['FR']['upper']['CURRENT_ANGLE']`
- Default: `0.1465` radians
- Relative: `absolute - 0.1465`

**Index 18**: `joint_pos[6]` = `BL_upper` relative to default
- Absolute: `config.SERVO_CONFIG['BL']['upper']['CURRENT_ANGLE']`
- Default: `0.1465` radians
- Relative: `absolute - 0.1465`

**Index 19**: `joint_pos[7]` = `BR_upper` relative to default
- Absolute: `config.SERVO_CONFIG['BR']['upper']['CURRENT_ANGLE']`
- Default: `-0.1465` radians
- Relative: `absolute - (-0.1465)`

**Index 20**: `joint_pos[8]` = `FL_lower` relative to default
- Absolute: `config.SERVO_CONFIG['FL']['lower']['CURRENT_ANGLE']`
- Default: `0.0` radians
- Relative: `absolute - 0.0`

**Index 21**: `joint_pos[9]` = `FR_lower` relative to default
- Absolute: `config.SERVO_CONFIG['FR']['lower']['CURRENT_ANGLE']`
- Default: `0.0` radians
- Relative: `absolute - 0.0`

**Index 22**: `joint_pos[10]` = `BL_lower` relative to default
- Absolute: `config.SERVO_CONFIG['BL']['lower']['CURRENT_ANGLE']`
- Default: `0.0` radians
- Relative: `absolute - 0.0`

**Index 23**: `joint_pos[11]` = `BR_lower` relative to default
- Absolute: `config.SERVO_CONFIG['BR']['lower']['CURRENT_ANGLE']`
- Default: `0.0` radians
- Relative: `absolute - 0.0`

---

### Index 24-35: `joint_vel` (12 values) - Joint Velocities
- **Units**: Radians/second (rad/s)
- **Frame**: Joint space
- **Description**: Rate of change of joint positions
- **Order**: Same as `joint_pos` (must match exactly)

**Index 24**: `joint_vel[0]` = `FL_hip` velocity
- Calculation: `(current_FL_hip - previous_FL_hip) / dt`
- `dt = 0.033` seconds (~30 Hz)

**Index 25**: `joint_vel[1]` = `FR_hip` velocity
- Calculation: `(current_FR_hip - previous_FR_hip) / dt`

**Index 26**: `joint_vel[2]` = `BL_hip` velocity
- Calculation: `(current_BL_hip - previous_BL_hip) / dt`

**Index 27**: `joint_vel[3]` = `BR_hip` velocity
- Calculation: `(current_BR_hip - previous_BR_hip) / dt`

**Index 28**: `joint_vel[4]` = `FL_upper` velocity
- Calculation: `(current_FL_upper - previous_FL_upper) / dt`

**Index 29**: `joint_vel[5]` = `FR_upper` velocity
- Calculation: `(current_FR_upper - previous_FR_upper) / dt`

**Index 30**: `joint_vel[6]` = `BL_upper` velocity
- Calculation: `(current_BL_upper - previous_BL_upper) / dt`

**Index 31**: `joint_vel[7]` = `BR_upper` velocity
- Calculation: `(current_BR_upper - previous_BR_upper) / dt`

**Index 32**: `joint_vel[8]` = `FL_lower` velocity
- Calculation: `(current_FL_lower - previous_FL_lower) / dt`

**Index 33**: `joint_vel[9]` = `FR_lower` velocity
- Calculation: `(current_FR_lower - previous_FR_lower) / dt`

**Index 34**: `joint_vel[10]` = `BL_lower` velocity
- Calculation: `(current_BL_lower - previous_BL_lower) / dt`

**Index 35**: `joint_vel[11]` = `BR_lower` velocity
- Calculation: `(current_BR_lower - previous_BR_lower) / dt`

---

### Index 36-47: `last_action` (12 values) - Previous Action Output
- **Units**: Unitless (normalized action in [-1, 1])
- **Frame**: Joint space
- **Description**: Raw model output from the previous timestep (before scaling to joint angles)
- **Order**: Same as `joint_pos` and `joint_vel` (must match exactly)
- **Range**: [-1, 1] (raw policy output)

**Index 36**: `last_action[0]` = Previous `FL_hip` action
- From: `result[0, 0]` from previous inference
- Range: [-1, 1]

**Index 37**: `last_action[1]` = Previous `FR_hip` action
- From: `result[0, 1]` from previous inference

**Index 38**: `last_action[2]` = Previous `BL_hip` action
- From: `result[0, 2]` from previous inference

**Index 39**: `last_action[3]` = Previous `BR_hip` action
- From: `result[0, 3]` from previous inference

**Index 40**: `last_action[4]` = Previous `FL_upper` action
- From: `result[0, 4]` from previous inference

**Index 41**: `last_action[5]` = Previous `FR_upper` action
- From: `result[0, 5]` from previous inference

**Index 42**: `last_action[6]` = Previous `BL_upper` action
- From: `result[0, 6]` from previous inference

**Index 43**: `last_action[7]` = Previous `BR_upper` action
- From: `result[0, 7]` from previous inference

**Index 44**: `last_action[8]` = Previous `FL_lower` action
- From: `result[0, 8]` from previous inference

**Index 45**: `last_action[9]` = Previous `FR_lower` action
- From: `result[0, 9]` from previous inference

**Index 46**: `last_action[10]` = Previous `BL_lower` action
- From: `result[0, 10]` from previous inference

**Index 47**: `last_action[11]` = Previous `BR_lower` action
- From: `result[0, 11]` from previous inference

---

## 12-Dimensional Action Vector (Output from Model)

The model outputs 12 values, each corresponding to a joint command. The order matches the observation's joint order exactly.

With `JOINT_ORDERING_SCHEME = "by_type"`:

**Action[0]**: `FL_hip` command
- Range: [-1, 1] (raw model output)
- Mapping: `target_angle = min_angle + (action + 1.0) * 0.5 * (max_angle - min_angle)`
- Applied to: `config.SERVO_CONFIG['FL']['hip']`

**Action[1]**: `FR_hip` command
- Range: [-1, 1]
- Applied to: `config.SERVO_CONFIG['FR']['hip']`

**Action[2]**: `BL_hip` command
- Range: [-1, 1]
- Applied to: `config.SERVO_CONFIG['BL']['hip']`

**Action[3]**: `BR_hip` command
- Range: [-1, 1]
- Applied to: `config.SERVO_CONFIG['BR']['hip']`

**Action[4]**: `FL_upper` command
- Range: [-1, 1]
- Applied to: `config.SERVO_CONFIG['FL']['upper']`

**Action[5]**: `FR_upper` command
- Range: [-1, 1]
- Applied to: `config.SERVO_CONFIG['FR']['upper']`

**Action[6]**: `BL_upper` command
- Range: [-1, 1]
- Applied to: `config.SERVO_CONFIG['BL']['upper']`

**Action[7]**: `BR_upper` command
- Range: [-1, 1]
- Applied to: `config.SERVO_CONFIG['BR']['upper']`

**Action[8]**: `FL_lower` command
- Range: [-1, 1]
- Applied to: `config.SERVO_CONFIG['FL']['lower']`

**Action[9]**: `FR_lower` command
- Range: [-1, 1]
- Applied to: `config.SERVO_CONFIG['FR']['lower']`

**Action[10]**: `BL_lower` command
- Range: [-1, 1]
- Applied to: `config.SERVO_CONFIG['BL']['lower']`

**Action[11]**: `BR_lower` command
- Range: [-1, 1]
- Applied to: `config.SERVO_CONFIG['BR']['lower']`

---

## Critical Ordering Consistency

The following must use the **exact same joint order**:

1. `joint_pos` (observation indices 12-23)
2. `joint_vel` (observation indices 24-35)
3. `last_action` (observation indices 36-47)
4. Model action output (indices 0-11)

**Current order (`"by_type"`)**:
```
[FL_hip, FR_hip, BL_hip, BR_hip, FL_upper, FR_upper, BL_upper, BR_upper, FL_lower, FR_lower, BL_lower, BR_lower]
```

This means:
- Observation `joint_pos[0]` = FL_hip position
- Observation `joint_vel[0]` = FL_hip velocity
- Observation `last_action[0]` = previous FL_hip action
- Model output `Action[0]` = new FL_hip command

**All four must align** for correct behavior.

---

## Summary Table

| Index Range | Component | Size | Units | Frame | Source |
|-------------|-----------|------|-------|--------|--------|
| 0-2 | base_lin_vel | 3 | m/s | World | Zeroed (no odometry) |
| 3-5 | base_ang_vel | 3 | rad/s | World | Gyroscope (roll, pitch, yaw) |
| 6-8 | projected_gravity | 3 | unitless | Body | Accelerometer (move, shift, translate) |
| 9-11 | velocity_commands | 3 | m/s, m/s, rad/s | Body | User commands (w/s/a/d/arrows) |
| 12-23 | joint_pos | 12 | rad (relative) | Joint | SERVO_CONFIG CURRENT_ANGLE - defaults |
| 24-35 | joint_vel | 12 | rad/s | Joint | Finite difference of joint_pos |
| 36-47 | last_action | 12 | [-1, 1] | Joint | Previous model output |

**Action output**: 12 values in [-1, 1], same order as joint observations.

---

## Joint Ordering Schemes

### "by_type" (Current)
Order: `[FL_hip, FR_hip, BL_hip, BR_hip, FL_upper, FR_upper, BL_upper, BR_upper, FL_lower, FR_lower, BL_lower, BR_lower]`

- Groups all hips together, then all uppers, then all lowers
- Matches Anymal's convention in Isaac Lab

### "by_leg" (Alternative)
Order: `[FL_hip, FL_upper, FL_lower, FR_hip, FR_upper, FR_lower, BL_hip, BL_upper, BL_lower, BR_hip, BR_upper, BR_lower]`

- Groups all joints for each leg together
- FL complete, then FR complete, then BL complete, then BR complete

**Note**: The ordering scheme is controlled by `config.JOINT_ORDERING_SCHEME` and must be consistent across all joint-related vectors.
