# Joint Ordering Analysis - Critical Mismatch Found!

## Problem
The physical robot legs move differently than simulated, suggesting a **joint ordering mismatch** between:
1. What the model expects (Isaac Lab's internal joint order)
2. What your code is providing (observation/action order)

---

## Isaac Lab's Joint Ordering Convention

### For Anymal (from symmetry code):
Isaac Lab uses this order internally:
```
[
    'LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA',  # All hips first
    'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE',  # All thighs second
    'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE'   # All knees third
]
```

**Pattern**: Joints grouped by **TYPE** (hip, thigh, knee), not by leg.

### For Robot Dog:
Your robot uses: `FL, FR, BL, BR` naming (not `LF, LH, RF, RH`).

**CRITICAL QUESTION**: Does Isaac Lab order robot_dog joints the same way as Anymal?

---

## Current Code Analysis

### Observation Order (in `inference.py`):
```python
for leg_id in ['FL', 'FR', 'BL', 'BR']:
    for joint_name in ['hip', 'upper', 'lower']:
        joint_pos_abs.append(...)
```

This produces: `[FL_hip, FL_upper, FL_lower, FR_hip, FR_upper, FR_lower, BL_hip, BL_upper, BL_lower, BR_hip, BR_upper, BR_lower]`

**Order**: Legs grouped together (FL all joints, then FR all joints, etc.)

### Action Order (in `inference.py`):
```python
action_idx = 0
for leg_id in ['FL', 'FR', 'BL', 'BR']:
    for joint_name in ['hip', 'upper', 'lower']:
        target_action = result[0, action_idx]
        action_idx += 1
```

This expects: `[FL_hip, FL_upper, FL_lower, FR_hip, FR_upper, FR_lower, BL_hip, BL_upper, BL_lower, BR_hip, BR_upper, BR_lower]`

**Order**: Same as observation (legs grouped together)

---

## Migration Guide Says:
```
joint_pos: [FL_hip, FL_upper, FL_lower, FR_hip, FR_upper, FR_lower, BL_hip, BL_upper, BL_lower, BR_hip, BR_upper, BR_lower]
```

This matches your current code, BUT **this might be wrong** if Isaac Lab actually orders joints by TYPE like Anymal!

---

## Potential Correct Order (if Isaac Lab groups by joint type):

If robot_dog follows the same pattern as Anymal, the order should be:
```
[
    FL_hip, FR_hip, BL_hip, BR_hip,      # All hips first
    FL_upper, FR_upper, BL_upper, BR_upper,  # All uppers second
    FL_lower, FR_lower, BL_lower, BR_lower   # All lowers third
]
```

**Order**: Joints grouped by TYPE (hip, upper, lower), not by leg.

---

## How to Verify the Correct Order

### Option 1: Check the Model's IO Descriptors
If you have access to the model's IO descriptor file (`.yaml`), it will list the exact joint names in order.

### Option 2: Test with Known Actions
1. Send action `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` (only first joint active)
2. See which physical joint moves
3. If it's FL_hip → current order is correct
4. If it's something else → order is wrong

### Option 3: Check Isaac Lab's Joint Ordering
The actual order Isaac Lab uses depends on:
- How the URDF defines joints
- How Isaac Lab reads/orders them internally
- The `joint_ids` configuration in the environment

---

## Recommended Fix

### If Isaac Lab groups by TYPE (like Anymal):

**Observation order should be**:
```python
# Group by joint type, not by leg
joint_pos_abs = []
for joint_name in ['hip', 'upper', 'lower']:
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        joint_pos_abs.append(float(config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE']))
```

**Action order should be**:
```python
# Group by joint type, not by leg
action_idx = 0
for joint_name in ['hip', 'upper', 'lower']:
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        target_action = result[0, action_idx]
        # ... apply to leg_id, joint_name
        action_idx += 1
```

---

## Next Steps

1. **Verify the actual joint order** the model expects (check IO descriptors or test)
2. **Update the migration guide** if the order is wrong
3. **Fix the code** to match the correct order
4. **Test** that physical robot matches simulation

---

## Action Output Order from Model

The model outputs actions in the same order it expects observations. So:
- If observation is `[FL_hip, FL_upper, FL_lower, FR_hip, ...]` → actions are `[FL_hip_action, FL_upper_action, ...]`
- If observation is `[FL_hip, FR_hip, BL_hip, BR_hip, FL_upper, ...]` → actions are `[FL_hip_action, FR_hip_action, ...]`

**The action order MUST match the observation order!**
