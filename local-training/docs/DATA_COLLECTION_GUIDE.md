# Data Collection Guide for Train Detection Model

## Current Status
- **Train Present**: 29 images
- **No Train**: 6 images  
- **Class Imbalance**: 4.8:1 ratio (needs improvement)

## Priority Collection Areas

### 1. Balance Dataset (Immediate Need)
- **Target**: Collect 50+ "no train" images to balance the 29 train images
- **Goal**: Achieve 1:1 or 2:1 ratio for better learning

### 2. Scenario Diversity

#### Lighting Conditions
- [ ] Dawn/dusk lighting
- [ ] Bright sunny conditions  
- [ ] Overcast/cloudy weather
- [ ] Night scenes with artificial lighting
- [ ] Shadows and backlighting

#### Train Types & Positions
- [ ] Different train types (passenger, freight, locomotives)
- [ ] Trains at different distances (close-up, medium, far)
- [ ] Multiple trains in one image
- [ ] Trains partially visible (entering/leaving frame)
- [ ] Different angles (side view, angled, front/back)

#### Environmental Variations
- [ ] Different seasons (summer, winter, fall, spring)
- [ ] Weather conditions (rain, snow, fog)
- [ ] Different track configurations (straight, curved, elevated)
- [ ] Urban vs rural settings
- [ ] Different camera heights/perspectives

#### Background Complexity
- [ ] Similar objects that could confuse the model:
  - Long vehicles (buses, trucks)
  - Buildings with train-like features
  - Industrial equipment
  - Bridges and overpasses

### 3. No-Train Scenarios

#### Essential No-Train Categories
- [ ] Empty railroad tracks (various angles)
- [ ] Railroad infrastructure without trains
- [ ] Similar transportation (buses, trucks, subway cars not on tracks)
- [ ] Industrial scenes that might be confused with trains
- [ ] General landscapes and urban scenes
- [ ] Railroad crossings with cars/pedestrians but no trains

## Quality Guidelines

### Image Quality Standards
- **Resolution**: Minimum 800x600, prefer 1024x768 or higher
- **Focus**: Clear, not blurry
- **Exposure**: Well-exposed, not too dark or bright
- **Format**: JPG, PNG (avoid heavily compressed images)

### Naming Convention
```
YYYYMMDD_HHMMSS_[category]_[description].jpg

Examples:
20250915_143022_train_present_freight_side_view.jpg
20250915_143105_no_train_empty_tracks_sunset.jpg
```

### Organization Structure
```
dataset/
├── train_present/
│   ├── freight/
│   ├── passenger/
│   ├── multiple/
│   └── partial/
├── no_train/
│   ├── empty_tracks/
│   ├── infrastructure/
│   ├── similar_objects/
│   └── general/
└── validation/
    ├── train_present/
    └── no_train/
```

## Data Collection Tips

### 1. Batch Collection
- Collect images in batches of 20-50
- Retrain model after each significant batch
- Compare performance improvements

### 2. Hard Negatives
- Pay special attention to images the model gets wrong
- Collect more examples similar to failure cases
- These improve model robustness

### 3. Geographic Diversity
- Collect from different locations/regions
- Different railroad companies/styles
- Various track configurations

### 4. Temporal Diversity
- Different times of day
- Different seasons
- Different weather conditions

## Next Collection Targets

### Phase 1 (Next 50 images)
- [ ] 30 "no train" images (balance dataset)
- [ ] 10 trains in different lighting
- [ ] 10 empty tracks in various conditions

### Phase 2 (Next 100 images)  
- [ ] 25 freight trains
- [ ] 25 passenger trains
- [ ] 25 challenging "no train" scenarios
- [ ] 25 partial/distant trains

### Phase 3 (Next 200 images)
- [ ] Seasonal variations
- [ ] Weather conditions
- [ ] Night scenes
- [ ] Complex backgrounds

## Validation Strategy
- Reserve 20% of new images for validation
- Never use validation images for training
- Track performance on this held-out set