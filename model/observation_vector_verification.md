# Observation Vector Verification

## 48-Dim Observation Vector Structure

The Isaac Lab model expects a **48-dimensional** observation vector in this exact order:

```
[base_lin_vel(3), base_ang_vel(3), projected_gravity(3), velocity_commands(3),
 joint_pos(12), joint_vel(12), last_action(12)]
```

### Component Breakdown

1. **base_lin_vel** (3 values): `[vx, vy, vz]` in m/s
   - Currently zeroed: `[0.0, 0.0, 0.0]`

2. **base_ang_vel** (3 values): `[wx, wy, wz]` in rad/s
   - From gyroscope: `[roll, pitch, yaw]` converted from deg/s to rad/s

3. **projected_gravity** (3 values): `[gx, gy, gz]` normalized unit vector
   - From accelerometer: `[move, shift, translate]` normalized

4. **velocity_commands** (3 values): `[lin_vel_x, lin_vel_y, ang_vel_z]`
   - From discrete commands (w/s/a/d/arrowleft/arrowright)
   - Range: lin_vel_x/y: [-0.6, 0.6] m/s, ang_vel_z: [-0.8, 0.8] rad/s

5. **joint_pos** (12 values): Joint positions relative to default positions (radians)
   - **Order depends on `JOINT_ORDERING_SCHEME`**:
     - `"by_type"`: `[FL_hip, FR_hip, BL_hip, BR_hip, FL_upper, FR_upper, BL_upper, BR_upper, FL_lower, FR_lower, BL_lower, BR_lower]`
     - `"by_leg"`: `[FL_hip, FL_upper, FL_lower, FR_hip, FR_upper, FR_lower, BL_hip, BL_upper, BL_lower, BR_hip, BR_upper, BR_lower]`

6. **joint_vel** (12 values): Joint velocities in rad/s
   - **MUST be in SAME ORDER as joint_pos**
   - Computed via finite difference: `(current_pos - prev_pos) / dt`

7. **last_action** (12 values): Previous action output in [-1, 1]
   - **MUST be in SAME ORDER as joint_pos**
   - Raw model output from previous step

---

## Critical Ordering Requirements

### ✅ All Joint-Related Vectors Must Match

The following three vectors **MUST** use the **exact same ordering**:
- `joint_pos` (observation input)
- `joint_vel` (observation input)  
- `last_action` (observation input)
- Model action output (used to update `last_action`)

### Why This Matters

1. **Observation Order**: The model was trained with joints in a specific order
2. **Action Order**: The model outputs actions in the same order it expects observations
3. **Consistency**: If `joint_pos` and `last_action` are in different orders, the model will misinterpret the relationship between current state and previous actions

---

## Current Configuration

- **JOINT_ORDERING_SCHEME**: `"by_type"` (set in `config.py`)
- **Order**: `[FL_hip, FR_hip, BL_hip, BR_hip, FL_upper, FR_upper, BL_upper, BR_upper, FL_lower, FR_lower, BL_lower, BR_lower]`

---

## Verification Steps

### 1. Check Observation Vector Shape
The code includes assertions to verify:
- Total size = 48
- Each component has correct size (3, 3, 3, 3, 12, 12, 12)

### 2. Verify Joint Ordering Consistency
All three joint vectors (`joint_pos`, `joint_vel`, `last_action`) are built using the same `JOINT_ORDERING_SCHEME`, ensuring consistency.

### 3. Test with Single Active Joint
Send action `[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` and verify:
- With `"by_type"`: Should move **FL_hip** (first joint)
- With `"by_leg"`: Should move **FL_hip** (first joint)

### 4. Compare with Simulation
Run the same command in Isaac Lab simulation and compare joint movements with physical robot.

---

## Code Flow Verification

1. **Observation Construction** (`inference.py`):
   - `joint_pos` built using `JOINT_ORDERING_SCHEME`
   - `joint_vel` built from `joint_pos_abs` (same order)
   - `last_action` loaded from `config.LAST_ACTION` (stored from previous step)

2. **Action Parsing** (`inference.py`):
   - Model output `result[0, :12]` parsed using same `JOINT_ORDERING_SCHEME`
   - Actions applied to correct joints

3. **State Storage** (`inference.py`):
   - `config.LAST_ACTION = result[0, :12]` (already in correct order)
   - `config.PREV_JOINT_POS_ABS = joint_pos_abs` (same order as current)

---

## Expected Behavior

With `JOINT_ORDERING_SCHEME = "by_type"`:

**Observation joint order**:
```
Index 0:  FL_hip
Index 1:  FR_hip
Index 2:  BL_hip
Index 3:  BR_hip
Index 4:  FL_upper
Index 5:  FR_upper
Index 6:  BL_upper
Index 7:  BR_upper
Index 8:  FL_lower
Index 9:  FR_lower
Index 10: BL_lower
Index 11: BR_lower
```

**Action output order** (from model):
- Same as observation order above
- Action[0] controls FL_hip, Action[1] controls FR_hip, etc.

---

## Troubleshooting

If physical robot moves incorrectly:

1. **Verify ordering scheme**: Check `config.JOINT_ORDERING_SCHEME` matches model's expected order
2. **Check debug logs**: Look for "Observation vector shape" log message
3. **Test single joint**: Send action with only one non-zero value
4. **Compare orders**: Ensure `joint_pos`, `joint_vel`, and `last_action` all use same order
5. **Try alternative**: Switch between `"by_type"` and `"by_leg"` and test

---

## Summary

✅ **Observation vector is correctly structured**:
- All components in correct order
- All joint vectors use same ordering scheme
- Size verification included
- Debug logging added

✅ **Action parsing matches observation order**:
- Uses same `JOINT_ORDERING_SCHEME`
- Actions applied to correct joints

✅ **State storage maintains consistency**:
- `LAST_ACTION` stored in correct order
- `PREV_JOINT_POS_ABS` stored in correct order

The code should now correctly match Isaac Lab's expected input format!
