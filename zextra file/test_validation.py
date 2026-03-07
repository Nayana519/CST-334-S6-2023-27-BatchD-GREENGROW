# -*- coding: utf-8 -*-
# Test script for crop-location validation

CROP_SUITABILITY = {
    'Maize': ['North India', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Himachal Pradesh', 'Central India', 'Madhya Pradesh', 'Karnataka'],
    'Rice': ['East India', 'West Bengal', 'Assam', 'Bihar', 'Jharkhand', 'South India', 'Karnataka', 'Tamil Nadu', 'Andhra Pradesh', 'Telangana', 'Kerala', 'Northeast'],
    'Wheat': ['North India', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Rajasthan', 'Madhya Pradesh', 'Delhi'],
    'Cotton': ['Central India', 'Maharashtra', 'Gujarat', 'Telangana', 'Karnataka', 'Andhra Pradesh', 'West India'],
    'Sugarcane': ['North India', 'Uttar Pradesh', 'Punjab', 'West India', 'Maharashtra', 'Gujarat', 'South India', 'Karnataka', 'Tamil Nadu', 'Kerala'],
    'Tobacco': ['South India', 'Karnataka', 'Andhra Pradesh', 'Telangana', 'Tamil Nadu', 'Central India'],
    'Barley': ['North India', 'Rajasthan', 'Uttar Pradesh', 'Haryana', 'Punjab', 'Himachal Pradesh'],
    'Oil seeds': ['Central India', 'Madhya Pradesh', 'Rajasthan', 'South India', 'Karnataka', 'Andhra Pradesh', 'Telangana', 'Maharashtra'],
    'Pulses': ['North India', 'Uttar Pradesh', 'Madhya Pradesh', 'Rajasthan', 'South India', 'Karnataka', 'Kerala'],
    'Ground Nuts': ['South India', 'Karnataka', 'Andhra Pradesh', 'Tamil Nadu', 'West India', 'Gujarat']
}

CITY_TO_STATE = {
    'delhi': 'Delhi', 'new delhi': 'Delhi',
    'punjab': 'Punjab', 'amritsar': 'Punjab', 'chandigarh': 'Chandigarh',
    'ludhiana': 'Punjab', 'jalandhar': 'Punjab',
    'maharashtra': 'Maharashtra', 'mumbai': 'Maharashtra',
    'kerala': 'Kerala', 'kochi': 'Kerala', 'thiruvananthapuram': 'Kerala',
    'karnataka': 'Karnataka', 'bangalore': 'Karnataka', 'bengaluru': 'Karnataka',
    'tamil nadu': 'Tamil Nadu', 'chennai': 'Tamil Nadu',
    'westbengal': 'West Bengal', 'kolkata': 'West Bengal',
}

def get_state_from_city(city):
    city_lower = city.lower().strip()
    if city_lower in CITY_TO_STATE:
        return CITY_TO_STATE[city_lower], True
    return "Unknown", False

def validate_crop_location(crop_name, state):
    if state not in CITY_TO_STATE.values():
        return True, []

    is_valid = state in CROP_SUITABILITY.get(crop_name, [])
    suitable_crops = []
    for crop, states in CROP_SUITABILITY.items():
        if state in states:
            suitable_crops.append(crop)

    return is_valid, suitable_crops

# Test cases
print("=" * 60)
print("CROP-LOCATION VALIDATION TEST SUITE")
print("=" * 60)

# Test 1: Cotton in Kerala (incompatible)
print("\n[TEST 1] Cotton in Kerala (should be INCOMPATIBLE)")
state, found = get_state_from_city("Kerala")
is_valid, crops = validate_crop_location("Cotton", state)
print("  Location found: " + str(found))
print("  State: " + state)
print("  Is valid: " + str(is_valid))
print("  Suitable crops for " + state + ": " + str(crops))
if not is_valid and len(crops) > 0:
    print("  [PASS] Test passed")
else:
    print("  [FAIL] Test failed")

# Test 2: Maize in Punjab (compatible)
print("\n[TEST 2] Maize in Punjab (should be COMPATIBLE)")
state, found = get_state_from_city("Punjab")
is_valid, crops = validate_crop_location("Maize", state)
print("  Location found: " + str(found))
print("  State: " + state)
print("  Is valid: " + str(is_valid))
if is_valid:
    print("  [PASS] Test passed")
else:
    print("  [FAIL] Test failed")

# Test 3: Unknown city
print("\n[TEST 3] Unknown city (should show location not found warning)")
state, found = get_state_from_city("RandomCity123")
is_valid, crops = validate_crop_location("Rice", state)
print("  Location found: " + str(found))
print("  State: " + state)
print("  Is valid: " + str(is_valid))
if not found and state == "Unknown":
    print("  [PASS] Test passed")
else:
    print("  [FAIL] Test failed")

# Test 4: Rice in West Bengal (compatible)
print("\n[TEST 4] Rice in Kolkata (should be COMPATIBLE)")
state, found = get_state_from_city("Kolkata")
is_valid, crops = validate_crop_location("Rice", state)
print("  Location found: " + str(found))
print("  State: " + state)
print("  Is valid: " + str(is_valid))
if is_valid:
    print("  [PASS] Test passed")
else:
    print("  [FAIL] Test failed")

# Test 5: Rice in Kerala (compatible)
print("\n[TEST 5] Rice in Kerala (should be COMPATIBLE)")
state, found = get_state_from_city("Kerala")
is_valid, crops = validate_crop_location("Rice", state)
print("  Location found: " + str(found))
print("  State: " + state)
print("  Is valid: " + str(is_valid))
print("  Suitable crops: " + str(crops))
if is_valid:
    print("  [PASS] Test passed")
else:
    print("  [FAIL] Test failed")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)
